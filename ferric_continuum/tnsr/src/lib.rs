#![allow(rustdoc::bare_urls)]
//! # tnsr — transformer autograd + checkpointing library
//!
//! Bazel-integrated Rust implementation of the per-device transformer math
//! taught in Ch.4 of the JAX scaling book
//! ("How To Scale Your Model", <https://jax-ml.github.io/scaling-book/>),
//! plus the gradient checkpointing / rematerialization strategies from Ch.5.
//! The default build is CPU-only and f32; selected forward kernels can use the
//! monorepo CUDA path when built with Bazel `--config=cuda`.
//!
//! See `SCALING_BOOK_MAP.md` at the crate root for the full chapter-by-chapter
//! mapping between book formulas and crate modules.
//!
//! The `scaling` module makes those formulas executable: it computes parameter
//! counts, FLOPs, activation bytes, roofline estimates, and sharding costs
//! directly from a `TransformerConfig`.  Its `scaling::distributed` sub-module
//! extends this to the multi-device mechanisms PyTorch uses — DDP, FSDP/ZeRO,
//! tensor and pipeline parallelism — as symbolic cost estimates plus runnable
//! single-process simulations over `Vec<f32>`.

pub(crate) mod attention_layout;
pub mod autograd;
pub mod bpe;
pub mod checkpoint;
pub mod cuda_ffi;
pub mod debug;
pub mod deepseek_v41;
pub mod dtensor;
pub mod grad_mode;
pub mod inference;
pub mod ops;
pub mod playground;
pub mod qwen3;
pub mod qwen3_load;
pub mod saved;
pub mod scaling;
pub mod tensor;
pub mod transformer;
pub mod typed;
