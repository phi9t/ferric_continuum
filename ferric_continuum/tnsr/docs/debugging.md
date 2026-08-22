# Debugging tnsr

This guide is for the Bazel-integrated `ferric_continuum/tnsr` module. It
adapts the useful standalone `tnsr` debugging notes to the current monorepo API
and target layout.

## Build and Run

Use the pinned Bazel binary first:

```sh
/data02/home/philip.yang/.local/bin/bazel-9.2.0 build //ferric_continuum/tnsr:tnsr
/data02/home/philip.yang/.local/bin/bazel-9.2.0 test //ferric_continuum/tnsr:lib_tests
/data02/home/philip.yang/.local/bin/bazel-9.2.0 run //ferric_continuum/tnsr:tnsr_demo -- --help
/data02/home/philip.yang/.local/bin/bazel-9.2.0 run //ferric_continuum/tnsr:tnsr_demo -- \
  --dot /tmp/tnsr/block.dot \
  --trace-json /tmp/tnsr/trace.json
```

Feature targets:

```sh
/data02/home/philip.yang/.local/bin/bazel-9.2.0 test //ferric_continuum/tnsr:debug_trace_tests
/data02/home/philip.yang/.local/bin/bazel-9.2.0 test //ferric_continuum/tnsr:inference_tests
/data02/home/philip.yang/.local/bin/bazel-9.2.0 test //ferric_continuum/tnsr:mesh_sim_tests
/data02/home/philip.yang/.local/bin/bazel-9.2.0 test //ferric_continuum/tnsr:scaling_tests
/data02/home/philip.yang/.local/bin/bazel-9.2.0 test //ferric_continuum/tnsr:scaling_memory_tests
/data02/home/philip.yang/.local/bin/bazel-9.2.0 test //ferric_continuum/tnsr:distributed_tests
/data02/home/philip.yang/.local/bin/bazel-9.2.0 test //ferric_continuum/tnsr:qwen3_tests
/data02/home/philip.yang/.local/bin/bazel-9.2.0 test //ferric_continuum/tnsr:context_parallel_gqa_tests
/data02/home/philip.yang/.local/bin/bazel-9.2.0 test //ferric_continuum/tnsr:playground_cli_tests
```

Cargo can be useful for editor-oriented local checks when the host environment
matches the crate metadata, but it is supplementary:

```sh
cd ferric_continuum/tnsr
cargo check
```

## Breakpoint Guide

Core forward path:

- `src/transformer.rs`: `TransformerBlock::forward`
- `src/ops/linear.rs`: `linear`
- `src/ops/attention.rs`: `attention_scores`, `causal_masked_softmax`,
  `attention_mix`
- `src/ops/rope.rs`: differentiable RoPE used by Qwen3-style blocks
- `src/ops/gqa.rs`: grouped-query attention path

Backward and saved tensors:

- `src/autograd.rs`: `Engine::new`, `Engine::backward`, `Engine::accumulate`
- `src/saved.rs`: `SavedTensor::save`, `SavedTensor::unpack`
- `src/checkpoint.rs`: `checkpoint`, `recompute_and_get`

Debug exports:

- `src/debug.rs`: `DebugRecorder::trace_json`,
  `DebugRecorder::trace_json_pretty`, `DebugRecorder::dot_string`
- `tests/debug_trace_test.rs`: stable `tnsr.debug_trace` schema expectations
- `src/playground.rs`: `tnsr_demo` argument parsing
- `src/main.rs`: `tnsr_demo` artifact generation and report output

Inference and Qwen3:

- `src/inference.rs`: `RopeConfig`, `apply_rope`, `KvCache`,
  `ContinuousBatcher`
- `src/qwen3.rs`: Qwen3 block/model wiring
- `src/qwen3_load.rs`: HuggingFace checkpoint loading and layout adaptation
- `src/bpe.rs`: minimal byte-level BPE tokenizer

Scaling and distributed reports:

- `src/scaling/report.rs`: graph-derived and formula scaling reports
- `src/scaling/memory.rs`: `training_memory_report` and ZeRO memory ladder
- `src/scaling/distributed/report.rs`: aggregate DDP/FSDP/TP/PP report
- `src/scaling/distributed/collectives.rs`: ring-model cost and simulations
- `src/scaling/distributed/tensor_parallel.rs`: column-then-row simulation

CUDA forward path:

- `src/cuda_ffi.rs`: feature-gated Rust FFI boundary
- `ferric_continuum/cuda_kernels`: CUDA C ABI targets used under
  `--config=cuda`
- `tests/cuda_forward_test.rs`: CPU/GPU forward agreement test; requires a
  CUDA GPU

## rust-lldb

The easiest LLDB path is to build the Bazel binary, then debug the output path
under `bazel-bin`:

```sh
/data02/home/philip.yang/.local/bin/bazel-9.2.0 build //ferric_continuum/tnsr:tnsr_demo
rust-lldb bazel-bin/ferric_continuum/tnsr/tnsr_demo
(lldb) breakpoint set --name tnsr::transformer::TransformerBlock::forward
(lldb) breakpoint set --name tnsr::autograd::Engine::backward
(lldb) breakpoint set --name tnsr::checkpoint::recompute_and_get
(lldb) breakpoint set --name tnsr::debug::DebugRecorder::trace_json
(lldb) run --dot /tmp/tnsr/block.dot --trace-json /tmp/tnsr/trace.json
```

To focus on a test binary, build the relevant target and use the path printed by
Bazel under `bazel-bin`:

```sh
/data02/home/philip.yang/.local/bin/bazel-9.2.0 build //ferric_continuum/tnsr:debug_trace_tests
rust-lldb bazel-bin/ferric_continuum/tnsr/debug_trace_tests
(lldb) breakpoint set --name tnsr::debug::DebugRecorder::trace_json
(lldb) run
```

## Artifact Checks

`tnsr_demo` writes generated files only when requested. A quick smoke should
inspect both artifact kinds:

```sh
rm -rf /tmp/tnsr-debug-smoke
/data02/home/philip.yang/.local/bin/bazel-9.2.0 run //ferric_continuum/tnsr:tnsr_demo -- \
  --dot /tmp/tnsr-debug-smoke/block.dot \
  --trace-json /tmp/tnsr-debug-smoke/trace.json
head -n 1 /tmp/tnsr-debug-smoke/block.dot
python3 -m json.tool /tmp/tnsr-debug-smoke/trace.json >/dev/null
```

Expected first-line DOT output starts with `digraph Autograd`. Expected JSON has
`schema` set to `tnsr.debug_trace`.
