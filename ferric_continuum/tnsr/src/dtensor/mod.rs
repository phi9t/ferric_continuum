//! Experimental DTensor-style mesh simulation substrate.
//!
//! This module is deliberately separate from the core local `Tensor` type while
//! the repo compares wrapper, tensor-native, and plan-first distributed
//! semantics.

pub mod harness;
pub mod layout;
pub mod mesh;
pub mod shard_map;

pub use layout::{Layout, LayoutError, Placement, ReduceOp};
pub use mesh::{MeshAxis, MeshError, ParallelDims5D, RankCoord5D};
pub use shard_map::{ShardError, ShardMap};
