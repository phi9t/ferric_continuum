# Foundation-Model Parallelism Resources

## Knowledge

- [Book: *How To Scale Your Model* — Google DeepMind](https://jax-ml.github.io/scaling-book/)
  Primary conceptual spine for `tnsr`; use for transformer accounting, rooflines, communication, training parallelism, inference, and GPU architecture.
- [Chapter: “How to Parallelize a Transformer for Training”](https://jax-ml.github.io/scaling-book/training/)
  Derives data, fully sharded data, tensor, and pipeline parallelism costs. Use alongside `tnsr/src/scaling/distributed/`.
- [Chapter: “All About Transformer Inference”](https://jax-ml.github.io/scaling-book/inference/)
  Separates prefill from generation and analyzes memory/compute tradeoffs. Use alongside `tnsr/src/scaling/inference.rs`.
- [Chapter: “How to Think About GPUs”](https://jax-ml.github.io/scaling-book/gpus/)
  Connects GPU rooflines and topology to LLM scaling. Use when moving from CUDA kernel behavior to multi-GPU design.
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
  Authoritative CUDA execution, memory, synchronization, and multi-GPU reference. Use for every claim about what a kernel or memory transfer does.
- [NVIDIA guide: CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
  Concise host/device model and launch overview. Use for understanding the `tnsr` Rust → C ABI → CUDA path.
- [PyTorch: Tensor Parallelism](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)
  Official reference for row-wise, column-wise, and sequence parallel styles. Use to compare real APIs with `tnsr`'s single-process simulations.
- [PyTorch: Fully Sharded Data Parallel](https://docs.pytorch.org/docs/stable/fsdp.html)
  Official FSDP semantics and constraints. Use alongside `tnsr/src/scaling/distributed/fsdp.rs`.
- [PyTorch: Pipeline Parallelism](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)
  Official stages, schedules, microbatches, and composition guidance. Use alongside `tnsr/src/scaling/distributed/pipeline.rs`.
- [Paper: “Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism”](https://arxiv.org/abs/1909.08053)
  Primary paper for intra-layer tensor parallelism in transformer blocks. Use after tracing `tnsr`'s Qwen3 MLP and attention projections.
- [Paper: “Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM”](https://arxiv.org/abs/2104.04473)
  Primary source on composing tensor, pipeline, and data parallelism. Use when designing a multi-dimensional experiment.
- [NVIDIA Megatron Core: Context Parallel Package](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
  Authoritative implementation guide for sequence-axis activation sharding, KV exchange, backward reduce-scatter, GQA benefits, and composition with TP/PP/DP.
- [Paper: “Ring Attention with Blockwise Transformers for Near-Infinite Context”](https://arxiv.org/abs/2310.01889)
  Primary source for circulating KV blocks while computing blockwise attention. Use when replacing a correctness-first KV all-gather with bounded-memory execution.
- [Paper: “DeepSpeed Ulysses”](https://arxiv.org/abs/2309.14509)
  Primary source for the alternative all-to-all sequence/head redistribution design. Use to compare ring KV exchange with head-parallel attention.

## Wisdom (Communities)

- [NVIDIA Developer Forums: CUDA Programming and Performance](https://forums.developer.nvidia.com/c/accelerated-computing/cuda/206)
  Practitioner feedback on kernel correctness, profiling, and hardware-specific behavior. Use after producing a minimal benchmark and profiler evidence.
- [PyTorch Forums: distributed category](https://discuss.pytorch.org/c/distributed/12)
  Framework-practitioner discussion of collective, FSDP, TP, and pipeline behavior. Use to test design assumptions against real deployments.

## Gaps

- A target GPU and interconnect topology have not been identified, so hardware-specific optimization sources are intentionally deferred.
- No preferred production comparison stack (for example PyTorch, JAX, or Megatron-Core) has been selected.
