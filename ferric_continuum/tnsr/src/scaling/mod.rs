//! Executable scaling math for transformer blocks.
//!
//! This module provides the analytical layer that maps the JAX scaling book's
//! FLOPs and memory formulas to the actual `TransformerBlock` and
//! `TransformerConfig` types in this crate.
//!
//! Book: "How To Scale Your Model", https://jax-ml.github.io/scaling-book/
//!
//! # Sub-modules
//!
//! | Module | Book chapter | What it computes |
//! |--------|-------------|------------------|
//! | `model_stats` | Ch.4 Transformer Accounting | param counts (QKVO, MLP, norm) |
//! | `op_cost` | Ch.4 FLOPs formulas | per-op fwd/bwd FLOPs + act bytes |
//! | `report` | Ch.4 summary | full block report + ASCII table |
//! | `roofline` | Ch.1 Roofline Analysis | compute-vs-memory bottleneck |
//! | `sharding` | Ch.3 Sharded Matrices | 4 sharding cases (symbolic) |
//! | `inference` | Ch.7 Inference | KV cache + peak activation bytes |
//!
//! # Quick start
//!
//! ```text
//! use tnsr::scaling::report::{scale_report, format_report};
//! use tnsr::transformer::TransformerConfig;
//!
//! let cfg = TransformerConfig::tiny_4_7_29();
//! let r = scale_report(&cfg);
//! println!("{}", format_report(&r));
//! ```

/// Bytes per f32 element — used throughout the scaling module for byte estimates.
pub(super) const F32_BYTES: u64 = 4;

pub mod inference;
pub mod model_stats;
pub mod op_cost;
pub mod report;
pub mod roofline;
pub mod sharding;
