use super::layout::Placement;
use super::mesh::MeshAxis;

pub const MESH_SIM_TRACE_SCHEMA: &str = "tnsr.mesh_sim_trace.v0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingPhase {
    Forward,
    Loss,
    Backward,
    Optimizer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectiveKind {
    AllReduce,
    AllGather,
    ReduceScatter,
    Broadcast,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshTrace {
    pub schema: &'static str,
    pub events: Vec<MeshTraceEvent>,
}

impl Default for MeshTrace {
    fn default() -> Self {
        Self {
            schema: MESH_SIM_TRACE_SCHEMA,
            events: Vec::new(),
        }
    }
}

impl MeshTrace {
    pub fn record(&mut self, event: MeshTraceEvent) {
        self.events.push(event);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshTraceEvent {
    LayoutTransition {
        phase: TrainingPhase,
        tensor: String,
        src: String,
        dst: String,
    },
    Collective {
        phase: TrainingPhase,
        kind: CollectiveKind,
        axis: MeshAxis,
        tensor: Option<String>,
        bytes: u64,
        ranks: Vec<usize>,
    },
    Injection {
        phase: TrainingPhase,
        label: String,
        rank: usize,
    },
    Failure(FailureRecord),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailureRecord {
    pub phase: TrainingPhase,
    pub message: String,
    pub rank: Option<usize>,
    pub axis: Option<MeshAxis>,
    pub tensor: Option<String>,
    pub layout: Option<String>,
    pub collective: Option<CollectiveKind>,
}

pub fn placement_label(placement: Placement) -> String {
    match placement {
        Placement::Replicate => "Replicate".to_string(),
        Placement::Shard(dim) => format!("Shard({dim})"),
        Placement::Partial(_) => "Partial(sum)".to_string(),
        Placement::Local => "Local".to_string(),
    }
}
