//! Dump a real `tnsr.mesh_sim_trace.v0` trace from the graduated DTensor mesh
//! simulation, as consumed by the distributed-training TLA+ trace-refinement
//! bridge (`formal/distributed_training/trace_bridge.py`, wayfinder issue 08).
//!
//! This is intentionally tiny: it runs the canonical tiny-dense mesh simulation
//! (optionally with a fault injection) and prints the Rust-owned JSON trace to
//! stdout. It is the "emit real trace JSON from Rust" path -- the bridge
//! consumes exactly what the runtime recorded, not a hand-written fixture.
//!
//! Usage:
//!   dtensor_trace_dump                 # legal step -> commit-path trace
//!   dtensor_trace_dump --inject-failure  # failing step -> failure event(s)

use tnsr::dtensor::{
    run_tiny_dense_mesh_simulation, CollectiveKind, InjectionEvent, InjectionKind, InjectionPlan,
    MeshAxis, ParallelDims5D, TrainingPhase,
};

fn main() {
    let inject_failure = std::env::args().any(|a| a == "--inject-failure");
    // Canonical tiny-dense mesh: 1 pp x 2 dp_replicate x 2 dp_shard x 1 cp x 2 tp.
    let dims = ParallelDims5D::new(1, 2, 2, 1, 2);

    let plan = if inject_failure {
        Some(InjectionPlan::new(vec![InjectionEvent {
            label: "trace_bridge_missing_rank".to_string(),
            phase: TrainingPhase::Backward,
            rank: 0,
            axis: Some(MeshAxis::DpReplicate),
            collective: Some(CollectiveKind::AllReduce),
            kind: InjectionKind::MissingParticipant,
        }]))
    } else {
        None
    };

    let report = match run_tiny_dense_mesh_simulation(dims, plan.as_ref()) {
        Ok(report) => report,
        Err(err) => {
            eprintln!("mesh simulation failed: {err:?}");
            std::process::exit(1);
        }
    };

    println!("{}", report.trace.trace_json_pretty());
}
