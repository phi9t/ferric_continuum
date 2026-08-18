//! Experimental DTensor-style mesh simulation substrate.
//!
//! This module is deliberately separate from the core local `Tensor` type while
//! the repo compares wrapper, tensor-native, and plan-first distributed
//! semantics.

pub mod harness;
pub mod mesh;

pub use mesh::{MeshAxis, MeshError, ParallelDims5D, RankCoord5D};
