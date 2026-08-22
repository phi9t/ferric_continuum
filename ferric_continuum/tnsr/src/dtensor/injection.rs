use crate::tensor::TensorValue;

use super::mesh::MeshAxis;
use super::trace::{CollectiveKind, FailureRecord, TrainingPhase};

#[derive(Debug, Clone, PartialEq)]
pub enum InjectionKind {
    MissingParticipant,
    CorruptShard { offset: usize, value: f32 },
    DelayRankGroup,
    MismatchedPlacement,
    InjectNan { offset: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct InjectionEvent {
    pub label: String,
    pub phase: TrainingPhase,
    pub rank: usize,
    pub axis: Option<MeshAxis>,
    pub collective: Option<CollectiveKind>,
    pub kind: InjectionKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InjectionPlan {
    pub events: Vec<InjectionEvent>,
}

impl InjectionPlan {
    pub fn new(events: Vec<InjectionEvent>) -> Self {
        Self { events }
    }

    pub fn events_for(
        &self,
        phase: TrainingPhase,
        rank: usize,
    ) -> impl Iterator<Item = &InjectionEvent> {
        self.events
            .iter()
            .filter(move |event| event.phase == phase && event.rank == rank)
    }

    pub fn matching_events(&self, phase: TrainingPhase, rank: usize) -> Vec<&InjectionEvent> {
        self.events_for(phase, rank).collect()
    }

    pub fn apply_to_shard(
        &self,
        phase: TrainingPhase,
        rank: usize,
        shard: &mut TensorValue,
    ) -> Vec<FailureRecord> {
        let mut failures = Vec::new();
        for event in self.events_for(phase, rank) {
            match &event.kind {
                InjectionKind::CorruptShard { offset, value } => {
                    if *offset < shard.data.len() {
                        shard.data_mut()[*offset] = *value;
                    }
                }
                InjectionKind::InjectNan { offset } => {
                    if *offset < shard.data.len() {
                        shard.data_mut()[*offset] = f32::NAN;
                    }
                }
                InjectionKind::MissingParticipant
                | InjectionKind::DelayRankGroup
                | InjectionKind::MismatchedPlacement => failures.push(FailureRecord {
                    phase,
                    message: event.label.clone(),
                    rank: Some(rank),
                    axis: event.axis,
                    tensor: None,
                    layout: None,
                    collective: event.collective,
                }),
            }
        }
        failures
    }
}
