use crate::tensor::{Shape, Tensor, TensorLayoutMeta, TensorValue};

use super::harness::TrainingStepScenario;
use super::injection::InjectionPlan;
use super::trace::{FailureRecord, MeshTrace};
use super::{
    CollectiveError, CollectiveSimulator, Layout, LayoutError, MeshAxis, MeshTraceEvent,
    ParallelDims5D, Placement, ReduceOp, ShardError, ShardMap, TrainingPhase,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeLayoutError {
    MissingLayoutMeta {
        rank: usize,
    },
    Layout(LayoutError),
    Shard(ShardError),
    Collective(CollectiveError),
    RankCountMismatch {
        expected: usize,
        got: usize,
    },
    RankMismatch {
        expected: usize,
        got: usize,
    },
    MetadataMismatch {
        rank: usize,
    },
    SimulatorDimsMismatch {
        metadata: ParallelDims5D,
        simulator: ParallelDims5D,
    },
    UnsupportedPropagation {
        src: String,
        dst: String,
    },
}

#[derive(Debug, Clone)]
pub struct NativeLayoutTracerBulletReport {
    pub reference_loss: f32,
    pub reference_output_shape: Shape,
    pub activation_full_shape: Shape,
    pub activation_matches_reference: bool,
    pub activation_tp_shard_shape: Shape,
    pub partial_dp_shard_shape: Shape,
    pub partial_replicated_shape: Shape,
    pub trace: MeshTrace,
    pub failures: Vec<FailureRecord>,
    pub loss_reduction_supported: bool,
}

impl From<LayoutError> for NativeLayoutError {
    fn from(err: LayoutError) -> Self {
        Self::Layout(err)
    }
}

impl From<ShardError> for NativeLayoutError {
    fn from(err: ShardError) -> Self {
        Self::Shard(err)
    }
}

impl From<CollectiveError> for NativeLayoutError {
    fn from(err: CollectiveError) -> Self {
        Self::Collective(err)
    }
}

pub fn redistribute_native_training_step_boundary(
    tensors: &[Tensor],
    dst: Layout,
    sim: &mut CollectiveSimulator,
    phase: TrainingPhase,
    tensor_name: &str,
) -> Result<Vec<Tensor>, NativeLayoutError> {
    let meta = validate_native_inputs(tensors)?;
    if meta.dims != sim.dims {
        return Err(NativeLayoutError::SimulatorDimsMismatch {
            metadata: meta.dims,
            simulator: sim.dims,
        });
    }
    if meta.layout == dst {
        return Ok(tensors.to_vec());
    }

    let (src_axis, src_placement) = require_single_axis(&meta.layout)?;
    let (dst_axis, dst_placement) = require_single_axis(&dst)?;
    let values: Vec<TensorValue> = tensors
        .iter()
        .map(|tensor| tensor.inner.borrow().value.clone())
        .collect();

    let out_values = match (src_placement, dst_placement) {
        (Placement::Replicate, Placement::Shard(dim)) if src_axis == dst_axis => {
            sim.local_slice(dst_axis, phase, tensor_name, dim, &values)?
        }
        (Placement::Shard(_), Placement::Replicate) if src_axis == dst_axis => {
            let _ = sim.all_gather(src_axis, phase, &values)?;
            let shard_map =
                ShardMap::new(meta.global_shape.clone(), meta.layout.clone(), meta.dims)?;
            let full = shard_map.reconstruct_from_rank_shards(&values)?;
            vec![full; meta.dims.world_size()]
        }
        (Placement::Partial(ReduceOp::Sum), Placement::Replicate) if src_axis == dst_axis => {
            sim.all_reduce_sum(src_axis, phase, &values)?
        }
        (Placement::Partial(ReduceOp::Sum), Placement::Shard(dim)) if src_axis == dst_axis => {
            sim.reduce_scatter_sum(src_axis, dim, phase, &values)?
        }
        _ => {
            return Err(NativeLayoutError::UnsupportedPropagation {
                src: layout_label(&meta.layout),
                dst: layout_label(&dst),
            });
        }
    };

    sim.trace.record(MeshTraceEvent::LayoutTransition {
        phase,
        tensor: tensor_name.to_string(),
        src: layout_label(&meta.layout),
        dst: layout_label(&dst),
    });

    Ok(out_values
        .into_iter()
        .enumerate()
        .map(|(rank, value)| {
            Tensor::from_value(value, tensors[rank].inner.borrow().autograd.requires_grad)
                .with_layout_meta(TensorLayoutMeta {
                    global_shape: meta.global_shape.clone(),
                    layout: dst.clone(),
                    dims: meta.dims,
                    rank,
                })
        })
        .collect())
}

fn validate_native_inputs(tensors: &[Tensor]) -> Result<TensorLayoutMeta, NativeLayoutError> {
    let first = tensors
        .first()
        .ok_or(NativeLayoutError::RankCountMismatch {
            expected: 1,
            got: 0,
        })?;
    let meta = first
        .layout_meta()
        .ok_or(NativeLayoutError::MissingLayoutMeta { rank: 0 })?;
    let expected = meta.dims.world_size();
    if tensors.len() != expected {
        return Err(NativeLayoutError::RankCountMismatch {
            expected,
            got: tensors.len(),
        });
    }
    for (rank, tensor) in tensors.iter().enumerate() {
        let Some(candidate) = tensor.layout_meta() else {
            return Err(NativeLayoutError::MissingLayoutMeta { rank });
        };
        if candidate.rank != rank {
            return Err(NativeLayoutError::RankMismatch {
                expected: rank,
                got: candidate.rank,
            });
        }
        if candidate.global_shape != meta.global_shape
            || candidate.layout != meta.layout
            || candidate.dims != meta.dims
        {
            return Err(NativeLayoutError::MetadataMismatch { rank });
        }
    }
    Ok(meta)
}

impl TrainingStepScenario {
    pub fn run_native_layout_tracer_bullet(
        &self,
        dims: super::ParallelDims5D,
        injection_plan: Option<&InjectionPlan>,
    ) -> Result<NativeLayoutTracerBulletReport, NativeLayoutError> {
        let reference = self.run_unsharded_reference();
        let mut sim = CollectiveSimulator::new(dims);

        let activation = self.reference_activation_value();
        let tp_replicated = Layout::new(vec![(MeshAxis::Tp, Placement::Replicate)])?;
        let tp_sharded = Layout::new(vec![(MeshAxis::Tp, Placement::Shard(2))])?;
        let activation_tensors = tensors_with_meta(
            &activation,
            activation.shape.clone(),
            tp_replicated.clone(),
            dims,
        );
        let activation_sharded = redistribute_native_training_step_boundary(
            &activation_tensors,
            tp_sharded,
            &mut sim,
            TrainingPhase::Forward,
            "native_activation",
        )?;
        let activation_tp_shard_shape = activation_sharded[0].shape();
        let activation_replicated = redistribute_native_training_step_boundary(
            &activation_sharded,
            tp_replicated,
            &mut sim,
            TrainingPhase::Forward,
            "native_activation",
        )?;
        let activation_full = activation_replicated[0].inner.borrow().value.clone();

        let dp_shard_partial =
            Layout::new(vec![(MeshAxis::DpShard, Placement::Partial(ReduceOp::Sum))])?;
        let dp_sharded = Layout::new(vec![(MeshAxis::DpShard, Placement::Shard(1))])?;
        let mut partial_tensors = Vec::with_capacity(dims.world_size());
        for rank in 0..dims.world_size() {
            let offset = rank as f32 * 100.0;
            let value = TensorValue::from_vec(
                Shape(vec![2, 4]),
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
            );
            partial_tensors.push(Tensor::from_value_no_grad(value).with_layout_meta(
                TensorLayoutMeta {
                    global_shape: Shape(vec![2, 4]),
                    layout: dp_shard_partial.clone(),
                    dims,
                    rank,
                },
            ));
        }

        let mut failures = Vec::new();
        if let Some(plan) = injection_plan {
            for (rank, tensor) in partial_tensors.iter_mut().enumerate() {
                let mut value = tensor.inner.borrow().value.clone();
                for event in plan.matching_events(TrainingPhase::Backward, rank) {
                    sim.trace.record(MeshTraceEvent::Injection {
                        phase: event.phase,
                        label: event.label.clone(),
                        rank: event.rank,
                    });
                }
                for mut record in plan.apply_to_shard(TrainingPhase::Backward, rank, &mut value) {
                    record.tensor = Some("native_partial".to_string());
                    sim.trace.record(MeshTraceEvent::Failure(record.clone()));
                    failures.push(record);
                }
                tensor.inner.borrow_mut().value = value;
            }
        }

        let partial_sharded = redistribute_native_training_step_boundary(
            &partial_tensors,
            dp_sharded,
            &mut sim,
            TrainingPhase::Backward,
            "native_partial",
        )?;
        let partial_dp_shard_shape = partial_sharded[0].shape();

        let dp_replicate_partial = Layout::new(vec![(
            MeshAxis::DpReplicate,
            Placement::Partial(ReduceOp::Sum),
        )])?;
        let dp_replicated = Layout::new(vec![(MeshAxis::DpReplicate, Placement::Replicate)])?;
        let partial_replicate_tensors: Vec<Tensor> = (0..dims.world_size())
            .map(|rank| {
                Tensor::from_value_no_grad(TensorValue::from_vec(
                    Shape(vec![2, 4]),
                    vec![rank as f32; 8],
                ))
                .with_layout_meta(TensorLayoutMeta {
                    global_shape: Shape(vec![2, 4]),
                    layout: dp_replicate_partial.clone(),
                    dims,
                    rank,
                })
            })
            .collect();
        let partial_replicated = redistribute_native_training_step_boundary(
            &partial_replicate_tensors,
            dp_replicated,
            &mut sim,
            TrainingPhase::Backward,
            "native_partial_replicate",
        )?;
        let partial_replicated_shape = partial_replicated[0].shape();

        Ok(NativeLayoutTracerBulletReport {
            reference_loss: reference.loss,
            reference_output_shape: reference.output_shape,
            activation_matches_reference: activation_full.shape == activation.shape
                && activation_full.data.as_ref() == activation.data.as_ref(),
            activation_full_shape: activation_full.shape,
            activation_tp_shard_shape,
            partial_dp_shard_shape,
            partial_replicated_shape,
            trace: sim.trace,
            failures,
            loss_reduction_supported: false,
        })
    }
}

fn tensors_with_meta(
    value: &TensorValue,
    global_shape: Shape,
    layout: Layout,
    dims: super::ParallelDims5D,
) -> Vec<Tensor> {
    (0..dims.world_size())
        .map(|rank| {
            Tensor::from_value_no_grad(value.clone()).with_layout_meta(TensorLayoutMeta {
                global_shape: global_shape.clone(),
                layout: layout.clone(),
                dims,
                rank,
            })
        })
        .collect()
}

fn require_single_axis(layout: &Layout) -> Result<(MeshAxis, Placement), NativeLayoutError> {
    match single_axis_placement(layout)? {
        Some(axis_placement) => Ok(axis_placement),
        None => Err(NativeLayoutError::UnsupportedPropagation {
            src: layout_label(layout),
            dst: "single-axis distributed layout".to_string(),
        }),
    }
}

fn single_axis_placement(
    layout: &Layout,
) -> Result<Option<(MeshAxis, Placement)>, NativeLayoutError> {
    if layout.axes.len() > 1 {
        return Err(NativeLayoutError::UnsupportedPropagation {
            src: layout_label(layout),
            dst: "single-axis layout".to_string(),
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
