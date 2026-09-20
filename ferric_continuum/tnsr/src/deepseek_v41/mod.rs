//! DeepSeek V4.1 model-reading and execution building blocks.
//!
//! Read this module as an executable derivation of the release surfaces that
//! are small enough to inspect in `tnsr`:
//!
//! - [`config`] reads the HuggingFace and converted inference metadata that
//!   decide which mechanisms are present.
//! - [`load`] maps safetensors names and upstream tensor layouts into the tnsr
//!   parameter layout used by the CPU verifier paths.
//! - [`model`] wires text blocks, optional multimodal image spans, Engram
//!   updates, and the DSpark `forward_spec` path.
//! - [`attention`], [`sparse`], and [`engram`] keep the CED, CSA2,
//!   sliding-window, indexer, and hashed n-gram mechanics visible as separate
//!   seams for fixture parity tests.
//! - [`vision`] and [`vision_grid`] cover the ViT/aligner and image token-span
//!   layout used by the multimodal path.
//!
//! The real-weight and native-quantized execution claims remain gated by the
//! compatibility scripts and their receipts; this module-level code is the
//! readable Rust candidate those scripts exercise.

pub mod attention;
pub mod checkpoint_io;
pub mod config;
pub mod cost;
pub mod dspark;
pub mod engram;
pub mod hc_tensor;
pub mod hyper;
pub mod load;
pub mod model;
pub mod moe;
pub mod residual;
pub mod rope;
pub mod sparse;
pub mod vision;
pub mod vision_grid;
