# Lesson 0001: The three layers of `tnsr`

**Time:** 8 minutes
**Tangible win:** Trace one Qwen3 projection to CUDA and explain why that does not make the model GPU-resident or distributed.

Your first research skill is to identify whether code is:

1. executing a model;
2. crossing onto a GPU; or
3. modeling parallelism without actually running it.

Those are three different kinds of evidence.

## Start with a prediction

Decide whether this statement is true before reading onward:

> Building `qwen3_infer` with `--config=cuda` runs Qwen3 inference on the GPU.

<details>
<summary>Reveal the answer</summary>

**Misleading.** Selected linear forwards use CUDA, but tensors live as host-side `Rc<Vec<f32>>` values. Each eligible call copies buffers to newly allocated device buffers and copies the result back.

Qwen3's fused GQA, norms, RoPE, activations, residuals, token selection, and every backward operation remain on the CPU.

</details>

## One repository, three research layers

| Layer | Relevant code | What it represents |
|---|---|---|
| Tensor and model engine | `tensor.rs`, `autograd.rs`, `ops/`, `qwen3.rs` | Real host-side model execution and gradient recording |
| CUDA enclave | `cuda_ffi.rs`, `cuda_kernels/` | Real GPU kernels reached through a host-buffer C ABI |
| Scaling laboratory | `scaling/` | FLOP/byte estimates and single-process shard simulations |

The distinction mirrors the central research question in Google DeepMind's [*How To Scale Your Model*](https://jax-ml.github.io/scaling-book/): useful scaling decisions connect model math to hardware and communication. In `tnsr`, those concerns exist, but they are not yet joined into one distributed runtime.

## Trace one Qwen3 projection

Follow the query projection:

1. `Qwen3Attention::forward` calls:

   ```rust
   linear::linear(x, &self.wq, "q_proj")
   ```

2. `linear::linear` delegates the numeric forward to `raw_linear_forward`.

3. `raw_linear_forward` checks `cuda_ffi::use_cuda()`. Bazel's CUDA setting becomes a Rust crate feature.

4. `cuda_ffi::gemm_f32` calls the C symbol `ferric_cuda_gemm_f32` using pointers to host slices.

5. `gemm.cu`:

   - allocates three device buffers;
   - copies A and B from host to device;
   - launches `GemmNaiveKernel`;
   - copies C from device to host.

6. The returned `Vec<f32>` becomes another host-side `TensorValue`. The next non-linear operation runs on the CPU.

This follows NVIDIA's official [CUDA host/device programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html). The research-relevant question is how often `tnsr` crosses that boundary.

> **Key inference:** Kernel acceleration and GPU residency are not synonyms. A fast kernel can lose to a CPU loop when allocation, launch, synchronization, and host/device transfer overhead dominate its useful work.

## The Qwen3 softmax exception

The generic transformer attention path calls `softmax_last_dim`, which is CUDA-eligible. Qwen3 does not use that path. It calls `gqa_attention`, a fused Rust loop that computes scores, causal masking, softmax, and value mixing on the CPU.

In a CUDA-enabled Qwen3 forward:

```text
Q/K/V projections   GPU kernels, then results return to host
Q/K norm + RoPE     CPU
fused GQA           CPU
output projection   GPU kernel, then result returns to host
SwiGLU projections  GPU kernels around CPU activations
backward            CPU everywhere
```

## What “distributed” means here

`src/scaling/distributed/` is a valuable laboratory, not a transport runtime. It expresses the invariants of DDP, FSDP/ZeRO, tensor parallelism, and pipeline parallelism as formulas and deterministic single-process simulations.

It launches no devices, threads, processes, or collectives.

That makes it useful for the mission's first phase: prove the algebra before paying the complexity cost of a real backend. Compare its column/row tensor-parallel split with the [official PyTorch tensor-parallel styles](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html) and the original [Megatron-LM paper](https://arxiv.org/abs/1909.08053).

## Retrieval practice

Answer each question before opening its answer.

<details>
<summary>1. A function returns one Vec&lt;f32&gt; per logical rank. What evidence would prove it used multiple GPUs?</summary>

A device/backend allocation, actual rank or process launch, and a real communication primitive or profiler trace. Multiple vectors alone prove only a simulation.

</details>

<details>
<summary>2. Why might CUDA GEMM make this Qwen3 implementation slower?</summary>

Every projection may pay for device allocation, two host-to-device copies, a launch and synchronization, then a device-to-host copy. Small GEMMs or frequent crossings may not amortize that overhead.

</details>

<details>
<summary>3. Which file should you inspect first to decide whether Qwen3 attention softmax runs on CUDA?</summary>

Start at `qwen3.rs` to identify the called attention operation, then inspect `ops/gqa.rs`. Searching only for a CUDA softmax kernel would give the wrong conclusion.

</details>

## Two-minute fieldwork

Run these searches from the repository root and reconstruct the call chain:

```bash
rg -n "gqa_attention|linear::linear" ferric_continuum/tnsr/src/qwen3.rs
rg -n "use_cuda|gemm_f32|softmax_f32" ferric_continuum/tnsr/src
rg -n "cudaMemcpy|<<<" ferric_continuum/cuda_kernels
```

Check your result against the [`tnsr` repository map](../reference/0001-tnsr-repository-map.md).

## Primary source

Read the short NVIDIA [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) section. As you read, label each part of `cuda_ffi.rs` and `gemm.cu` as:

- host orchestration;
- memory movement; or
- device execution.

## Before the next lesson

Continue to [Lesson 0002: Context-parallel Qwen3](0002-context-parallel-qwen3.md), but first answer this from memory:

> Which parts of a CUDA-enabled Qwen3 forward cross the GPU boundary, and why could that path still be slower than CPU execution?

Ask follow-up questions whenever a boundary, term, or claim is unclear. Your answers and questions determine the next lesson's difficulty.
