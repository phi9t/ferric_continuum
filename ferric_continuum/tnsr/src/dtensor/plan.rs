use std::collections::HashMap;

use crate::tensor::{Shape, TensorValue};

use super::harness::TrainingStepScenario;
use super::injection::{InjectionEvent, InjectionPlan};
use super::trace::{CollectiveKind, FailureRecord, MeshTrace, MeshTraceEvent, TrainingPhase};
use super::{
    CollectiveError, CollectiveSimulator, Layout, LayoutError, MeshAxis, ParallelDims5D, Placement,
    ReduceOp, ShardError, ShardMap,
};

pub fn run_tiny_dense_mesh_simulation(
    dims: ParallelDims5D,
    injections: Option<&InjectionPlan>,
) -> Result<MeshSimPlanReport, PlanError> {
    MeshSimPlan::tiny_dense_training(dims)?.execute(injections)
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshSimPlan {
    pub dims: ParallelDims5D,
    pub steps: Vec<MeshSimStep>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MeshSimStep {
    Tensor {
        name: String,
        global_shape: Shape,
        layout: Layout,
    },
    Redistribute {
        tensor: String,
        src: Layout,
        dst: Layout,
        phase: TrainingPhase,
    },
    Collective {
        kind: CollectiveKind,
        axis: MeshAxis,
        tensor: String,
        phase: TrainingPhase,
    },
    Cost {
        tensor: String,
        phase: TrainingPhase,
        collective: CollectiveKind,
        axis: MeshAxis,
        bytes: u64,
    },
    Injection {
        event: InjectionEvent,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshSimCollectiveRecord {
    pub kind: CollectiveKind,
    pub axis: MeshAxis,
    pub tensor: String,
    pub phase: TrainingPhase,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshSimCost {
    pub tensor: String,
    pub phase: TrainingPhase,
    pub collective: CollectiveKind,
    pub axis: MeshAxis,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshSimPlanReport {
    pub trace: MeshTrace,
    pub stable_vocabulary: MeshSimStableVocabulary,
    pub dualpipe_schedule: DualPipeSchedule,
    pub planned_collectives: usize,
    pub executed_collectives: usize,
    pub planned_collective_records: Vec<MeshSimCollectiveRecord>,
    pub executed_collective_records: Vec<MeshSimCollectiveRecord>,
    pub costs: Vec<MeshSimCost>,
    pub failures: Vec<FailureRecord>,
    pub reference_loss: f32,
    pub reference_output_shape: Shape,
    pub activation_full_shape: Shape,
    pub activation_matches_reference: bool,
    pub activation_tp_shard_shape: Shape,
    pub partial_dp_shard_shape: Shape,
    pub partial_replicated_shape: Shape,
    pub loss_reduction_supported: bool,
}

impl MeshSimPlanReport {
    pub fn deterministic_run_report(&self) -> String {
        let mut lines = Vec::new();
        lines.push("entrypoint: run_tiny_dense_mesh_simulation".to_string());
        lines.push(format!("trace_schema: {}", self.trace.schema));
        lines.push(format!(
            "mesh_axes: {}",
            self.stable_vocabulary.mesh_axes.join(" | ")
        ));
        lines.push(format!(
            "rank_ordering: {}",
            self.stable_vocabulary.rank_coordinate_ordering
        ));
        for component_line in self.dualpipe_schedule.chunk_report_lines() {
            lines.push(component_line);
        }
        for record in &self.planned_collective_records {
            lines.push(format!(
                "collective: {} {} {} {}",
                phase_label(record.phase),
                collective_label(record.kind),
                axis_label(record.axis),
                record.tensor
            ));
        }
        for cost in &self.costs {
            lines.push(format!(
                "cost: {} {} {} {} {}",
                phase_label(cost.phase),
                collective_label(cost.collective),
                axis_label(cost.axis),
                cost.tensor,
                cost.bytes
            ));
        }
        lines.push(format!("planned_collectives: {}", self.planned_collectives));
        lines.push(format!(
            "executed_collectives: {}",
            self.executed_collectives
        ));
        lines.push(format!("failures: {}", self.failures.len()));
        lines.join("\n")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshSimStableVocabulary {
    pub mesh_axes: Vec<String>,
    pub rank_coordinate_ordering: String,
    pub process_group_construction: Vec<String>,
    pub placement_names: Vec<String>,
    pub layout_transition_names: Vec<String>,
    pub collective_signatures: Vec<String>,
    pub trace_event_fields: Vec<String>,
    pub failure_record_fields: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DualPipeChunk {
    Forward(usize),
    Backward(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DualPipeComponentKind {
    Attention,
    AllToAllDispatch,
    Mlp,
    AllToAllCombine,
    PpCommunication,
    AttentionInputGradient,
    AttentionWeightGradient,
    AllToAllDispatchGradient,
    MlpInputGradient,
    MlpWeightGradient,
    AllToAllCombineGradient,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DualPipeComponent {
    pub chunk: DualPipeChunk,
    pub kind: DualPipeComponentKind,
    pub depends_on: Vec<DualPipeComponentKind>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DualPipeOverlap {
    pub forward_chunk: DualPipeChunk,
    pub forward_component: DualPipeComponentKind,
    pub backward_chunk: DualPipeChunk,
    pub backward_component: DualPipeComponentKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DualPipeSchedule {
    pub components: Vec<DualPipeComponent>,
    overlaps: Vec<DualPipeOverlap>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    Collective(CollectiveError),
    Layout(LayoutError),
    Shard(ShardError),
    Schedule(FailureRecord),
    MissingTensor { name: String },
    UnsupportedStep { message: String },
}

#[derive(Clone)]
struct PlannedTensor {
    global_shape: Shape,
    layout: Layout,
    shards: Vec<TensorValue>,
}

impl From<CollectiveError> for PlanError {
    fn from(err: CollectiveError) -> Self {
        Self::Collective(err)
    }
}

impl From<LayoutError> for PlanError {
    fn from(err: LayoutError) -> Self {
        Self::Layout(err)
    }
}

impl From<ShardError> for PlanError {
    fn from(err: ShardError) -> Self {
        Self::Shard(err)
    }
}

impl MeshSimPlan {
    pub fn tiny_dense_training(dims: ParallelDims5D) -> Result<Self, PlanError> {
        let tp_replicated = Layout::new(vec![(MeshAxis::Tp, Placement::Replicate)])?;
        let tp_sharded = Layout::new(vec![(MeshAxis::Tp, Placement::Shard(2))])?;
        let dp_shard_partial =
            Layout::new(vec![(MeshAxis::DpShard, Placement::Partial(ReduceOp::Sum))])?;
        let dp_sharded = Layout::new(vec![(MeshAxis::DpShard, Placement::Shard(1))])?;
        let dp_replicate_partial = Layout::new(vec![(
            MeshAxis::DpReplicate,
            Placement::Partial(ReduceOp::Sum),
        )])?;
        let dp_replicated = Layout::new(vec![(MeshAxis::DpReplicate, Placement::Replicate)])?;

        Ok(Self {
            dims,
            steps: vec![
                MeshSimStep::Tensor {
                    name: "plan_activation".to_string(),
                    global_shape: Shape(vec![2, 4, 8]),
                    layout: tp_replicated.clone(),
                },
                MeshSimStep::Redistribute {
                    tensor: "plan_activation".to_string(),
                    src: tp_replicated.clone(),
                    dst: tp_sharded.clone(),
                    phase: TrainingPhase::Forward,
                },
                MeshSimStep::Redistribute {
                    tensor: "plan_activation".to_string(),
                    src: tp_sharded,
                    dst: tp_replicated,
                    phase: TrainingPhase::Forward,
                },
                MeshSimStep::Collective {
                    kind: CollectiveKind::AllGather,
                    axis: MeshAxis::Tp,
                    tensor: "plan_activation".to_string(),
                    phase: TrainingPhase::Forward,
                },
                MeshSimStep::Cost {
                    tensor: "plan_activation".to_string(),
                    phase: TrainingPhase::Forward,
                    collective: CollectiveKind::AllGather,
                    axis: MeshAxis::Tp,
                    bytes: 256,
                },
                MeshSimStep::Tensor {
                    name: "plan_grad_partial".to_string(),
                    global_shape: Shape(vec![2, 4]),
                    layout: dp_shard_partial.clone(),
                },
                MeshSimStep::Redistribute {
                    tensor: "plan_grad_partial".to_string(),
                    src: dp_shard_partial,
                    dst: dp_sharded,
                    phase: TrainingPhase::Backward,
                },
                MeshSimStep::Collective {
                    kind: CollectiveKind::ReduceScatter,
                    axis: MeshAxis::DpShard,
                    tensor: "plan_grad_partial".to_string(),
                    phase: TrainingPhase::Backward,
                },
                MeshSimStep::Cost {
                    tensor: "plan_grad_partial".to_string(),
                    phase: TrainingPhase::Backward,
                    collective: CollectiveKind::ReduceScatter,
                    axis: MeshAxis::DpShard,
                    bytes: 32,
                },
                MeshSimStep::Tensor {
                    name: "plan_ddp_partial".to_string(),
                    global_shape: Shape(vec![2, 4]),
                    layout: dp_replicate_partial.clone(),
                },
                MeshSimStep::Redistribute {
                    tensor: "plan_ddp_partial".to_string(),
                    src: dp_replicate_partial,
                    dst: dp_replicated,
                    phase: TrainingPhase::Backward,
                },
                MeshSimStep::Collective {
                    kind: CollectiveKind::AllReduce,
                    axis: MeshAxis::DpReplicate,
                    tensor: "plan_ddp_partial".to_string(),
                    phase: TrainingPhase::Backward,
                },
                MeshSimStep::Cost {
                    tensor: "plan_ddp_partial".to_string(),
                    phase: TrainingPhase::Backward,
                    collective: CollectiveKind::AllReduce,
                    axis: MeshAxis::DpReplicate,
                    bytes: 64,
                },
            ],
        })
    }

    pub fn execute(
        &self,
        injections: Option<&InjectionPlan>,
    ) -> Result<MeshSimPlanReport, PlanError> {
        let dualpipe_schedule = DualPipeSchedule::tiny_dense_training();
        dualpipe_schedule.validate()?;
        let scenario = TrainingStepScenario::tiny_dense();
        let reference = scenario.run_unsharded_reference();
        let mut sim = CollectiveSimulator::new(self.dims);
        let mut tensors = HashMap::new();
        let mut failures = Vec::new();
        let mut activation_full = None;
        let mut activation_tp_shard_shape = None;
        let mut partial_dp_shard_shape = None;
        let mut partial_replicated_shape = None;

        for step in &self.steps {
            match step {
                MeshSimStep::Tensor {
                    name,
                    global_shape,
                    layout,
                } => {
                    let shards = initial_tensor_shards(name, global_shape, layout, self.dims)?;
                    tensors.insert(
                        name.clone(),
                        PlannedTensor {
                            global_shape: global_shape.clone(),
                            layout: layout.clone(),
                            shards,
                        },
                    );
                }
                MeshSimStep::Redistribute {
                    tensor,
                    src,
                    dst,
                    phase,
                } => {
                    apply_matching_injections(
                        injections,
                        *phase,
                        tensor,
                        &mut tensors,
                        &mut sim,
                        &mut failures,
                    )?;
                    let next = redistribute_tensor(
                        tensor,
                        src,
                        dst,
                        *phase,
                        self.dims,
                        &mut sim,
                        &mut tensors,
                    )?;
                    if tensor == "plan_activation" && dst.placement(MeshAxis::Tp).is_some() {
                        if dst.placement(MeshAxis::Tp) == Some(Placement::Shard(2)) {
                            activation_tp_shard_shape = Some(next.shards[0].shape.clone());
                        }
                        if dst.placement(MeshAxis::Tp) == Some(Placement::Replicate) {
                            activation_full = Some(next.shards[0].clone());
                        }
                    }
                    if tensor == "plan_grad_partial" {
                        partial_dp_shard_shape = Some(next.shards[0].shape.clone());
                    }
                    if tensor == "plan_ddp_partial" {
                        partial_replicated_shape = Some(next.shards[0].shape.clone());
                    }
                    tensors.insert(tensor.clone(), next);
                }
                MeshSimStep::Collective { .. } | MeshSimStep::Cost { .. } => {}
                MeshSimStep::Injection { event } => {
                    apply_injection_event(
                        event,
                        default_injection_tensor(event),
                        &mut tensors,
                        &mut sim,
                        &mut failures,
                    )?;
                }
            }
        }

        let planned_collective_records = self.planned_collective_records();
        let executed_collective_records = MeshSimCollectiveRecord::from_trace(&sim.trace);
        let costs = self.costs();
        let activation = scenario.reference_activation_value();
        let activation_full = activation_full.ok_or_else(|| PlanError::MissingTensor {
            name: "plan_activation".to_string(),
        })?;

        Ok(MeshSimPlanReport {
            trace: sim.trace,
            stable_vocabulary: MeshSimStableVocabulary::for_plan_first(self.dims),
            dualpipe_schedule,
            planned_collectives: planned_collective_records.len(),
            executed_collectives: executed_collective_records.len(),
            planned_collective_records,
            executed_collective_records,
            costs,
            failures,
            reference_loss: reference.loss,
            reference_output_shape: reference.output_shape,
            activation_matches_reference: activation_full.shape == activation.shape
                && activation_full.data.as_ref() == activation.data.as_ref(),
            activation_full_shape: activation_full.shape,
            activation_tp_shard_shape: activation_tp_shard_shape.ok_or_else(|| {
                PlanError::MissingTensor {
                    name: "plan_activation_tp_shard".to_string(),
                }
            })?,
            partial_dp_shard_shape: partial_dp_shard_shape.ok_or_else(|| {
                PlanError::MissingTensor {
                    name: "plan_grad_partial".to_string(),
                }
            })?,
            partial_replicated_shape: partial_replicated_shape.ok_or_else(|| {
                PlanError::MissingTensor {
                    name: "plan_ddp_partial".to_string(),
                }
            })?,
            loss_reduction_supported: false,
        })
    }

    pub fn planned_collective_records(&self) -> Vec<MeshSimCollectiveRecord> {
        self.steps
            .iter()
            .filter_map(|step| match step {
                MeshSimStep::Collective {
                    kind,
                    axis,
                    tensor,
                    phase,
                } => Some(MeshSimCollectiveRecord {
                    kind: *kind,
                    axis: *axis,
                    tensor: tensor.clone(),
                    phase: *phase,
                }),
                _ => None,
            })
            .collect()
    }

    pub fn costs(&self) -> Vec<MeshSimCost> {
        self.steps
            .iter()
            .filter_map(|step| match step {
                MeshSimStep::Cost {
                    tensor,
                    phase,
                    collective,
                    axis,
                    bytes,
                } => Some(MeshSimCost {
                    tensor: tensor.clone(),
                    phase: *phase,
                    collective: *collective,
                    axis: *axis,
                    bytes: *bytes,
                }),
                _ => None,
            })
            .collect()
    }
}

impl MeshSimStableVocabulary {
    pub fn for_plan_first(dims: ParallelDims5D) -> Self {
        Self {
            mesh_axes: vec![
                "pp,dp_replicate,dp_shard,cp,tp".to_string(),
                format!(
                    "sizes={},{},{},{},{}",
                    dims.pp, dims.dp_replicate, dims.dp_shard, dims.cp, dims.tp
                ),
            ],
            rank_coordinate_ordering:
                "row-major: pp -> dp_replicate -> dp_shard -> cp -> tp; tp is fastest".to_string(),
            process_group_construction: vec![
                "groups are built per axis by fixing all other coordinates".to_string(),
                "ordered membership is ascending global rank order".to_string(),
            ],
            placement_names: vec![
                "Replicate".to_string(),
                "Shard(dim)".to_string(),
                "Partial(sum)".to_string(),
                "Local".to_string(),
            ],
            layout_transition_names: vec![
                "Partial(sum)->Replicate".to_string(),
                "Partial(sum)->Shard(dim)".to_string(),
                "Shard(dim)->Replicate".to_string(),
                "Replicate->Shard(dim)".to_string(),
                "same-layout-noop".to_string(),
            ],
            collective_signatures: vec![
                "all_reduce_sum(axis, phase, shards)".to_string(),
                "all_gather(axis, phase, shards)".to_string(),
                "reduce_scatter_sum(axis, shard_dim, phase, shards)".to_string(),
                "broadcast(axis, phase, root_axis_index, shards)".to_string(),
            ],
            trace_event_fields: vec![
                "LayoutTransition{phase,tensor,src,dst}".to_string(),
                "Collective{phase,kind,axis,tensor,bytes,ranks}".to_string(),
                "Injection{phase,label,rank}".to_string(),
                "Failure(FailureRecord)".to_string(),
            ],
            failure_record_fields: vec![
                "phase".to_string(),
                "message".to_string(),
                "rank".to_string(),
                "axis".to_string(),
                "tensor".to_string(),
                "layout".to_string(),
                "collective".to_string(),
            ],
        }
    }
}

impl DualPipeSchedule {
    pub fn tiny_dense_training() -> Self {
        let forward = [
            DualPipeComponentKind::Attention,
            DualPipeComponentKind::AllToAllDispatch,
            DualPipeComponentKind::Mlp,
            DualPipeComponentKind::AllToAllCombine,
            DualPipeComponentKind::PpCommunication,
        ];
        let backward = [
            DualPipeComponentKind::AttentionInputGradient,
            DualPipeComponentKind::AttentionWeightGradient,
            DualPipeComponentKind::AllToAllDispatchGradient,
            DualPipeComponentKind::MlpInputGradient,
            DualPipeComponentKind::MlpWeightGradient,
            DualPipeComponentKind::AllToAllCombineGradient,
            DualPipeComponentKind::PpCommunication,
        ];
        let mut components = Vec::with_capacity(forward.len() + backward.len());
        for (idx, kind) in forward.iter().copied().enumerate() {
            components.push(DualPipeComponent {
                chunk: DualPipeChunk::Forward(0),
                kind,
                depends_on: if idx == 0 {
                    Vec::new()
                } else {
                    vec![forward[idx - 1]]
                },
            });
        }
        for (idx, kind) in backward.iter().copied().enumerate() {
            components.push(DualPipeComponent {
                chunk: DualPipeChunk::Backward(0),
                kind,
                depends_on: if idx == 0 {
                    Vec::new()
                } else {
                    vec![backward[idx - 1]]
                },
            });
        }
        Self {
            components,
            overlaps: vec![DualPipeOverlap {
                forward_chunk: DualPipeChunk::Forward(0),
                forward_component: DualPipeComponentKind::Mlp,
                backward_chunk: DualPipeChunk::Backward(0),
                backward_component: DualPipeComponentKind::AttentionInputGradient,
            }],
        }
    }

    pub fn component_kinds(&self, chunk: DualPipeChunk) -> Vec<DualPipeComponentKind> {
        self.components
            .iter()
            .filter_map(|component| {
                if component.chunk == chunk {
                    Some(component.kind)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn add_overlap(
        &mut self,
        first_chunk: DualPipeChunk,
        first_component: DualPipeComponentKind,
        second_chunk: DualPipeChunk,
        second_component: DualPipeComponentKind,
    ) {
        self.overlaps.push(DualPipeOverlap {
            forward_chunk: first_chunk,
            forward_component: first_component,
            backward_chunk: second_chunk,
            backward_component: second_component,
        });
    }

    pub fn legal_overlaps(&self) -> &[DualPipeOverlap] {
        &self.overlaps
    }

    pub fn validate(&self) -> Result<(), PlanError> {
        for component in &self.components {
            for dependency in &component.depends_on {
                if !self.component_kinds(component.chunk).contains(dependency) {
                    return Err(PlanError::Schedule(schedule_failure(
                        component.chunk,
                        component.kind,
                        Some(*dependency),
                        "dependency is not present in the same chunk",
                    )));
                }
            }
        }

        for overlap in &self.overlaps {
            let forward_is_forward = matches!(overlap.forward_chunk, DualPipeChunk::Forward(_));
            let backward_is_backward = matches!(overlap.backward_chunk, DualPipeChunk::Backward(_));
            if !(forward_is_forward && backward_is_backward) {
                return Err(PlanError::Schedule(schedule_failure(
                    overlap.backward_chunk,
                    overlap.backward_component,
                    Some(overlap.forward_component),
                    "overlap crosses an explicit same-direction dependency boundary",
                )));
            }
            if !self
                .component_kinds(overlap.forward_chunk)
                .contains(&overlap.forward_component)
                || !self
                    .component_kinds(overlap.backward_chunk)
                    .contains(&overlap.backward_component)
            {
                return Err(PlanError::Schedule(schedule_failure(
                    overlap.backward_chunk,
                    overlap.backward_component,
                    Some(overlap.forward_component),
                    "overlap references a component outside the schedule",
                )));
            }
        }
        Ok(())
    }

    fn chunk_report_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for chunk in [DualPipeChunk::Forward(0), DualPipeChunk::Backward(0)] {
            let labels = self
                .component_kinds(chunk)
                .into_iter()
                .map(component_label)
                .collect::<Vec<_>>()
                .join(" -> ");
            lines.push(format!("{}: {labels}", chunk_label(chunk)));
        }
        for overlap in &self.overlaps {
            lines.push(format!(
                "dualpipe.overlap: {}.{} || {}.{}",
                chunk_label(overlap.forward_chunk),
                component_label(overlap.forward_component),
                chunk_label(overlap.backward_chunk),
                component_label(overlap.backward_component)
            ));
        }
        lines
    }
}

impl MeshSimCollectiveRecord {
    pub fn from_trace(trace: &MeshTrace) -> Vec<Self> {
        trace
            .events
            .iter()
            .filter_map(|event| match event {
                MeshTraceEvent::Collective {
                    phase,
                    kind,
                    axis,
                    tensor: Some(tensor),
                    ..
                } => Some(Self {
                    kind: *kind,
                    axis: *axis,
                    tensor: tensor.clone(),
                    phase: *phase,
                }),
                _ => None,
            })
            .collect()
    }
}

fn initial_tensor_shards(
    name: &str,
    global_shape: &Shape,
    layout: &Layout,
    dims: ParallelDims5D,
) -> Result<Vec<TensorValue>, PlanError> {
    let value = match name {
        "plan_activation" => TrainingStepScenario::tiny_dense().reference_activation_value(),
        "plan_grad_partial" => {
            return Ok((0..dims.world_size())
                .map(|rank| {
                    let offset = rank as f32 * 100.0;
                    TensorValue::from_vec(
                        global_shape.clone(),
                        vec![
                            1.0 + offset,
                            2.0 + offset,
                            3.0 + offset,
                            4.0 + offset,
                            10.0 + offset,
                            20.0 + offset,
                            30.0 + offset,
                            40.0 + offset,
                        ],
                    )
                })
                .collect());
        }
        "plan_ddp_partial" => {
            return Ok((0..dims.world_size())
                .map(|rank| TensorValue::from_vec(global_shape.clone(), vec![rank as f32; 8]))
                .collect());
        }
        other => {
            return Err(PlanError::UnsupportedStep {
                message: format!("no tensor initializer for {other}"),
            });
        }
    };

    match single_axis_placement(layout)? {
        None | Some((_, Placement::Replicate)) => Ok(vec![value; dims.world_size()]),
        Some((_, Placement::Shard(_))) => {
            let shard_map = ShardMap::new(global_shape.clone(), layout.clone(), dims)?;
            (0..dims.world_size())
                .map(|rank| {
                    shard_map
                        .shard_tensor_for_rank(&value, rank)
                        .map_err(Into::into)
                })
                .collect()
        }
        Some((_, Placement::Partial(ReduceOp::Sum))) | Some((_, Placement::Local)) => {
            Err(PlanError::UnsupportedStep {
                message: format!(
                    "unsupported initial layout for {name}: {}",
                    layout_label(layout)
                ),
            })
        }
    }
}

fn redistribute_tensor(
    tensor: &str,
    src: &Layout,
    dst: &Layout,
    phase: TrainingPhase,
    dims: ParallelDims5D,
    sim: &mut CollectiveSimulator,
    tensors: &mut HashMap<String, PlannedTensor>,
) -> Result<PlannedTensor, PlanError> {
    let current = tensors
        .get(tensor)
        .ok_or_else(|| PlanError::MissingTensor {
            name: tensor.to_string(),
        })?;
    if current.layout != *src {
        return Err(PlanError::UnsupportedStep {
            message: format!(
                "plan expected {tensor} layout {} but runtime has {}",
                layout_label(src),
                layout_label(&current.layout)
            ),
        });
    }
    if src == dst {
        return Ok(current.clone());
    }

    let (src_axis, src_placement) = require_single_axis(src)?;
    let (dst_axis, dst_placement) = require_single_axis(dst)?;
    let shards = match (src_placement, dst_placement) {
        (Placement::Replicate, Placement::Shard(dim)) if src_axis == dst_axis => {
            sim.local_slice(dst_axis, phase, tensor, dim, &current.shards)?
        }
        (Placement::Shard(_), Placement::Replicate) if src_axis == dst_axis => {
            let _ = sim.all_gather(src_axis, phase, &current.shards)?;
            annotate_last_collective(&mut sim.trace, CollectiveKind::AllGather, src_axis, tensor);
            let shard_map = ShardMap::new(current.global_shape.clone(), src.clone(), dims)?;
            let full = shard_map.reconstruct_from_rank_shards(&current.shards)?;
            vec![full; dims.world_size()]
        }
        (Placement::Partial(ReduceOp::Sum), Placement::Replicate) if src_axis == dst_axis => {
            let out = sim.all_reduce_sum(src_axis, phase, &current.shards)?;
            annotate_last_collective(&mut sim.trace, CollectiveKind::AllReduce, src_axis, tensor);
            out
        }
        (Placement::Partial(ReduceOp::Sum), Placement::Shard(dim)) if src_axis == dst_axis => {
            let out = sim.reduce_scatter_sum(src_axis, dim, phase, &current.shards)?;
            annotate_last_collective(
                &mut sim.trace,
                CollectiveKind::ReduceScatter,
                src_axis,
                tensor,
            );
            out
        }
        _ => {
            return Err(PlanError::UnsupportedStep {
                message: format!(
                    "unsupported redistribution for {tensor}: {} -> {}",
                    layout_label(src),
                    layout_label(dst)
                ),
            });
        }
    };

    sim.trace.record(MeshTraceEvent::LayoutTransition {
        phase,
        tensor: tensor.to_string(),
        src: layout_label(src),
        dst: layout_label(dst),
    });

    Ok(PlannedTensor {
        global_shape: current.global_shape.clone(),
        layout: dst.clone(),
        shards,
    })
}

fn apply_matching_injections(
    injection_plan: Option<&InjectionPlan>,
    phase: TrainingPhase,
    tensor: &str,
    tensors: &mut HashMap<String, PlannedTensor>,
    sim: &mut CollectiveSimulator,
    failures: &mut Vec<FailureRecord>,
) -> Result<(), PlanError> {
    let Some(injection_plan) = injection_plan else {
        return Ok(());
    };
    for rank in 0..sim.dims.world_size() {
        for event in injection_plan.matching_events(phase, rank) {
            apply_injection_event(event, tensor, tensors, sim, failures)?;
        }
    }
    Ok(())
}

fn apply_injection_event(
    event: &InjectionEvent,
    tensor: &str,
    tensors: &mut HashMap<String, PlannedTensor>,
    sim: &mut CollectiveSimulator,
    failures: &mut Vec<FailureRecord>,
) -> Result<(), PlanError> {
    let planned = tensors
        .get_mut(tensor)
        .ok_or_else(|| PlanError::MissingTensor {
            name: tensor.to_string(),
        })?;
    let shard = planned
        .shards
        .get_mut(event.rank)
        .ok_or_else(|| PlanError::UnsupportedStep {
            message: format!("injection rank {} is outside {tensor}", event.rank),
        })?;
    sim.trace.record(MeshTraceEvent::Injection {
        phase: event.phase,
        label: event.label.clone(),
        rank: event.rank,
    });
    let one_event_plan = InjectionPlan::new(vec![event.clone()]);
    for mut record in one_event_plan.apply_to_shard(event.phase, event.rank, shard) {
        record.tensor = Some(tensor.to_string());
        record.layout = Some(layout_label(&planned.layout));
        sim.trace.record(MeshTraceEvent::Failure(record.clone()));
        failures.push(record);
    }
    Ok(())
}

fn annotate_last_collective(
    trace: &mut MeshTrace,
    expected_kind: CollectiveKind,
    expected_axis: MeshAxis,
    tensor: &str,
) {
    if let Some(MeshTraceEvent::Collective {
        kind,
        axis,
        tensor: trace_tensor,
        ..
    }) = trace.events.last_mut()
    {
        debug_assert_eq!(*kind, expected_kind);
        debug_assert_eq!(*axis, expected_axis);
        *trace_tensor = Some(tensor.to_string());
    }
}

fn default_injection_tensor(event: &InjectionEvent) -> &'static str {
    match event.collective {
        Some(CollectiveKind::ReduceScatter) => "plan_grad_partial",
        Some(CollectiveKind::AllReduce) => "plan_ddp_partial",
        Some(CollectiveKind::AllGather) | Some(CollectiveKind::Broadcast) | None => {
            "plan_activation"
        }
    }
}

fn require_single_axis(layout: &Layout) -> Result<(MeshAxis, Placement), PlanError> {
    match single_axis_placement(layout)? {
        Some(axis_placement) => Ok(axis_placement),
        None => Err(PlanError::UnsupportedStep {
            message: format!("expected single-axis layout, got {}", layout_label(layout)),
        }),
    }
}

fn single_axis_placement(layout: &Layout) -> Result<Option<(MeshAxis, Placement)>, PlanError> {
    if layout.axes.len() > 1 {
        return Err(PlanError::UnsupportedStep {
            message: format!("multi-axis layout unsupported: {}", layout_label(layout)),
        });
    }
    Ok(layout.axes.first().copied())
}

fn layout_label(layout: &Layout) -> String {
    if layout.axes.is_empty() {
        return "[]".to_string();
    }
    layout
        .axes
        .iter()
        .map(|(axis, placement)| {
            let placement = match placement {
                Placement::Replicate => "Replicate".to_string(),
                Placement::Shard(dim) => format!("Shard({dim})"),
                Placement::Partial(ReduceOp::Sum) => "Partial(sum)".to_string(),
                Placement::Local => "Local".to_string(),
            };
            format!("{axis:?}:{placement}")
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn schedule_failure(
    chunk: DualPipeChunk,
    component: DualPipeComponentKind,
    dependency: Option<DualPipeComponentKind>,
    message: &str,
) -> FailureRecord {
    FailureRecord {
        phase: chunk_phase(chunk),
        message: format!(
            "{message}: {}{}",
            component_label(component),
            dependency
                .map(|kind| format!(" depends_on {}", component_label(kind)))
                .unwrap_or_default()
        ),
        rank: None,
        axis: None,
        tensor: Some(format!(
            "{}.{}",
            chunk_label(chunk),
            component_label(component)
        )),
        layout: dependency.map(|kind| format!("{}.{}", chunk_label(chunk), component_label(kind))),
        collective: None,
    }
}

fn chunk_phase(chunk: DualPipeChunk) -> TrainingPhase {
    match chunk {
        DualPipeChunk::Forward(_) => TrainingPhase::Forward,
        DualPipeChunk::Backward(_) => TrainingPhase::Backward,
    }
}

fn chunk_label(chunk: DualPipeChunk) -> String {
    match chunk {
        DualPipeChunk::Forward(index) => format!("dualpipe.forward[{index}]"),
        DualPipeChunk::Backward(index) => format!("dualpipe.backward[{index}]"),
    }
}

fn component_label(kind: DualPipeComponentKind) -> &'static str {
    match kind {
        DualPipeComponentKind::Attention => "attention",
        DualPipeComponentKind::AllToAllDispatch => "all_to_all_dispatch",
        DualPipeComponentKind::Mlp => "mlp",
        DualPipeComponentKind::AllToAllCombine => "all_to_all_combine",
        DualPipeComponentKind::PpCommunication => "pp_communication",
        DualPipeComponentKind::AttentionInputGradient => "attention_input_gradient",
        DualPipeComponentKind::AttentionWeightGradient => "attention_weight_gradient",
        DualPipeComponentKind::AllToAllDispatchGradient => "all_to_all_dispatch_gradient",
        DualPipeComponentKind::MlpInputGradient => "mlp_input_gradient",
        DualPipeComponentKind::MlpWeightGradient => "mlp_weight_gradient",
        DualPipeComponentKind::AllToAllCombineGradient => "all_to_all_combine_gradient",
    }
}

fn phase_label(phase: TrainingPhase) -> &'static str {
    match phase {
        TrainingPhase::Forward => "forward",
        TrainingPhase::Loss => "loss",
        TrainingPhase::Backward => "backward",
        TrainingPhase::Optimizer => "optimizer",
    }
}

fn collective_label(kind: CollectiveKind) -> &'static str {
    match kind {
        CollectiveKind::AllReduce => "all_reduce",
        CollectiveKind::AllGather => "all_gather",
        CollectiveKind::ReduceScatter => "reduce_scatter",
        CollectiveKind::Broadcast => "broadcast",
    }
}

fn axis_label(axis: MeshAxis) -> &'static str {
    match axis {
        MeshAxis::Pp => "pp",
        MeshAxis::DpReplicate => "dp_replicate",
        MeshAxis::DpShard => "dp_shard",
        MeshAxis::Cp => "cp",
        MeshAxis::Tp => "tp",
    }
}
