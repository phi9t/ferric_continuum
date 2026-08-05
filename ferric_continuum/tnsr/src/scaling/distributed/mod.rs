//! Distributed-training mechanisms as executable estimates + single-process sims.
//!
//! Book reference: Ch.5 "Parallelize a Transformer for Training",
//! <https://jax-ml.github.io/scaling-book/training/>.
//!
//! `tnsr` is CPU-only, single-threaded, and f32 — it never launches a device or
//! opens a socket. This module teaches the *distributed* mechanisms PyTorch uses
//! in the only faithful way a single process can:
//!
//! 1. **Symbolic cost estimates** (same pattern as [`super::sharding`]):
//!    per-device FLOPs, memory, and collective bytes under the ring model.
//! 2. **Runnable simulations over `Vec<f32>`**: each mechanism is executed by
//!    looping over `D` logical shards in one process, so its correctness is
//!    provable — ring all-reduce equals a naive sum, column-then-row tensor
//!    parallel equals the unsharded matmul.
//!
//! # Concept → PyTorch → book chapter
//!
//! | Concept | PyTorch counterpart | Book | `tnsr` module |
//! |---------|---------------------|------|---------------|
//! | Collectives | `torch.distributed.{all_reduce,all_gather,reduce_scatter,broadcast}` | Ch.5 | [`collectives`] |
//! | Device mesh | `torch.distributed.device_mesh.DeviceMesh` | Ch.5 | [`mesh`] |
//! | Data parallel | `torch.nn.parallel.DistributedDataParallel` (DDP) | Ch.5 | [`data_parallel`] |
//! | ZeRO / FSDP | `torch.distributed.fsdp.FullyShardedDataParallel` | Ch.5 | [`fsdp`] |
//! | Tensor parallel | `torch.distributed.tensor.parallel` (Megatron) | Ch.3, Ch.5 | [`tensor_parallel`] |
//! | Pipeline parallel | `torch.distributed.pipelining` | Ch.5 | [`pipeline`] |
//! | Aggregate report | — | Ch.5 | [`report`] |
//!
//! `all-to-all` (expert / sequence parallelism) is out of scope; see the note in
//! [`collectives`].

pub mod collectives;
pub mod data_parallel;
pub mod fsdp;
pub mod mesh;
pub mod pipeline;
pub mod report;
pub mod tensor_parallel;
