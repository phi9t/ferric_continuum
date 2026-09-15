use super::layout::Placement;
use super::mesh::MeshAxis;

pub const MESH_SIM_TRACE_SCHEMA: &str = "tnsr.mesh_sim_trace.v0";

fn training_phase_label(phase: TrainingPhase) -> &'static str {
    match phase {
        TrainingPhase::Forward => "forward",
        TrainingPhase::Loss => "loss",
        TrainingPhase::Backward => "backward",
        TrainingPhase::Optimizer => "optimizer",
    }
}

fn collective_kind_label(kind: CollectiveKind) -> &'static str {
    match kind {
        CollectiveKind::AllReduce => "all_reduce",
        CollectiveKind::AllGather => "all_gather",
        CollectiveKind::ReduceScatter => "reduce_scatter",
        CollectiveKind::Broadcast => "broadcast",
    }
}

fn mesh_axis_label(axis: MeshAxis) -> &'static str {
    match axis {
        MeshAxis::Pp => "pp",
        MeshAxis::DpReplicate => "dp_replicate",
        MeshAxis::DpShard => "dp_shard",
        MeshAxis::Cp => "cp",
        MeshAxis::Tp => "tp",
    }
}

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

    /// Export a stable, Rust-owned JSON view of the recorded trace. This is the
    /// `tnsr.mesh_sim_trace.v0` schema consumed by the distributed-training
    /// TLA+ trace-refinement bridge (StepTxn.tla). The event stream is emitted
    /// in recorded order so a downstream checker can replay it as a behavior.
    pub fn trace_json(&self) -> serde_json::Value {
        serde_json::json!({
            "schema": self.schema,
            "events": self.events.iter().map(MeshTraceEvent::to_json).collect::<Vec<_>>(),
        })
    }

    pub fn trace_json_pretty(&self) -> String {
        serde_json::to_string_pretty(&self.trace_json()).expect("mesh sim trace JSON serialization")
    }
}

impl MeshTraceEvent {
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            MeshTraceEvent::LayoutTransition {
                phase,
                tensor,
                src,
                dst,
            } => serde_json::json!({
                "kind": "layout_transition",
                "phase": training_phase_label(*phase),
                "tensor": tensor,
                "src": src,
                "dst": dst,
            }),
            MeshTraceEvent::Collective {
                phase,
                kind,
                axis,
                tensor,
                bytes,
                ranks,
            } => serde_json::json!({
                "kind": "collective",
                "phase": training_phase_label(*phase),
                "collective": collective_kind_label(*kind),
                "axis": mesh_axis_label(*axis),
                "tensor": tensor,
                "bytes": bytes,
                "ranks": ranks,
            }),
            MeshTraceEvent::Injection { phase, label, rank } => serde_json::json!({
                "kind": "injection",
                "phase": training_phase_label(*phase),
                "label": label,
                "rank": rank,
            }),
            MeshTraceEvent::Failure(record) => serde_json::json!({
                "kind": "failure",
                "phase": training_phase_label(record.phase),
                "message": record.message,
                "rank": record.rank,
                "axis": record.axis.map(mesh_axis_label),
                "tensor": record.tensor,
                "layout": record.layout,
                "collective": record.collective.map(collective_kind_label),
            }),
        }
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
