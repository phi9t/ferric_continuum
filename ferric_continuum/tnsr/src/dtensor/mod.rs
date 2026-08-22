//! Experimental DTensor-style mesh simulation substrate.
//!
//! This module is deliberately separate from the core local `Tensor` type while
//! the repo compares wrapper, tensor-native, and plan-first distributed
//! semantics.

pub mod collective;
pub mod harness;
pub mod injection;
pub mod layout;
pub mod mesh;
pub mod native;
pub mod plan;
pub mod shard_map;
pub mod trace;
pub mod wrapper;

pub use crate::tensor::TensorLayoutMeta;
pub use collective::{CollectiveError, CollectiveSimulator};
pub use injection::{InjectionEvent, InjectionKind, InjectionPlan};
pub use layout::{Layout, LayoutError, Placement, ReduceOp};
pub use mesh::{MeshAxis, MeshError, ParallelDims5D, RankCoord5D};
pub use native::{
    redistribute_native_training_step_boundary, NativeLayoutError, NativeLayoutTracerBulletReport,
};
pub use plan::{
    run_tiny_dense_mesh_simulation, DualPipeChunk, DualPipeComponent, DualPipeComponentKind,
    DualPipeOverlap, DualPipeSchedule, MeshSimCollectiveRecord, MeshSimCost, MeshSimPlan,
    MeshSimPlanReport, MeshSimStableVocabulary, MeshSimStep, PlanError,
};
pub use shard_map::{ShardError, ShardMap};
pub use trace::{
    placement_label, CollectiveKind, FailureRecord, MeshTrace, MeshTraceEvent, TrainingPhase,
    MESH_SIM_TRACE_SCHEMA,
};
pub use wrapper::{DTensor, DtError, WrapperDtensorTracerBulletReport};
