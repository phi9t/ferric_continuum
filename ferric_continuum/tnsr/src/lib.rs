#![allow(rustdoc::bare_urls)]
//! # tnsr — transformer autograd + checkpointing library
//!
//! CPU-only, f32, no-external-deps Rust implementation of the per-device
//! transformer math taught in Ch.4 of the JAX scaling book
//! ("How To Scale Your Model", <https://jax-ml.github.io/scaling-book/>),
//! plus the gradient checkpointing / rematerialization strategies from Ch.5.
//!
//! See `SCALING_BOOK_MAP.md` at the crate root for the full chapter-by-chapter
//! mapping between book formulas and crate modules.
//!
//! The `scaling` module makes those formulas executable: it computes parameter
//! counts, FLOPs, activation bytes, roofline estimates, and sharding costs
//! directly from a `TransformerConfig`.

pub mod autograd;
pub mod checkpoint;
pub mod cuda_ffi;
pub mod debug;
pub mod grad_mode;
pub mod ops;
pub mod qwen3;
pub mod saved;
pub mod scaling;
pub mod tensor;
pub mod transformer;
