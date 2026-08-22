use crate::tensor::{Shape, TensorValue};
use std::fmt;

use super::harness::TrainingStepScenario;
use super::injection::InjectionPlan;
use super::trace::{FailureRecord, MeshTrace};
use super::{
    CollectiveError, CollectiveSimulator, Layout, LayoutError, MeshAxis, MeshTraceEvent,
    ParallelDims5D, Placement, ReduceOp, ShardError, ShardMap, TrainingPhase,
};

#[derive(Clone)]
pub struct DTensor {
    pub global_shape: Shape,
    pub layout: Layout,
    pub dims: ParallelDims5D,
    pub shards: Vec<TensorValue>,
}

impl fmt::Debug for DTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DTensor")
            .field("global_shape", &self.global_shape)
            .field("layout", &self.layout)
            .field("dims", &self.dims)
            .field(
                "shard_shapes",
                &self.shards.iter().map(|s| &s.shape).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl PartialEq for DTensor {
    fn eq(&self, other: &Self) -> bool {
        self.global_shape == other.global_shape
            && self.layout == other.layout
            && self.dims == other.dims
            && self.shards.len() == other.shards.len()
            && self
                .shards
                .iter()
                .zip(&other.shards)
                .all(|(lhs, rhs)| lhs.shape == rhs.shape && lhs.data.as_ref() == rhs.data.as_ref())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DtError {
    Layout(LayoutError),
    Shard(ShardError),
    Collective(CollectiveError),
    UnsupportedRedistribution { src: String, dst: String },
    RankCountMismatch { expected: usize, got: usize },
}

#[derive(Debug, Clone)]
pub struct WrapperDtensorTracerBulletReport {
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

impl From<LayoutError> for DtError {
    fn from(err: LayoutError) -> Self {
        Self::Layout(err)
    }
}

impl From<ShardError> for DtError {
    fn from(err: ShardError) -> Self {
        Self::Shard(err)
    }
}

impl From<CollectiveError> for DtError {
    fn from(err: CollectiveError) -> Self {
        Self::Collective(err)
    }
}

impl DTensor {
    pub fn from_replicated(
        value: TensorValue,
        layout: Layout,
        dims: ParallelDims5D,
    ) -> Result<Self, DtError> {
        let global_shape = value.shape.clone();
        let shards = match single_axis_placement(&layout)? {
            None | Some((_, Placement::Replicate)) => vec![value; dims.world_size()],
            Some((_, Placement::Partial(ReduceOp::Sum))) => {
                return Err(DtError::UnsupportedRedistribution {
                    src: "Replicate".to_string(),
                    dst: layout_label(&layout),
                });
            }
            Some((_, Placement::Shard(_))) => {
                let shard_map = ShardMap::new(global_shape.clone(), layout.clone(), dims)?;
                (0..dims.world_size())
                    .map(|rank| shard_map.shard_tensor_for_rank(&value, rank))
                    .collect::<Result<Vec<_>, _>>()?
            }
            Some((_, Placement::Local)) => {
                return Err(DtError::UnsupportedRedistribution {
                    src: "Replicate".to_string(),
                    dst: layout_label(&layout),
                });
            }
        };
        Ok(Self {
            global_shape,
            layout,
            dims,
            shards,
        })
    }

    pub fn redistribute(
        &self,
        dst: Layout,
        sim: &mut CollectiveSimulator,
        phase: TrainingPhase,
    ) -> Result<Self, DtError> {
        self.validate_rank_count()?;
        if self.layout == dst {
            return Ok(self.clone());
        }

        let (src_axis, src_placement) = require_single_axis(&self.layout)?;
        let (dst_axis, dst_placement) = require_single_axis(&dst)?;
        let shards = match (src_placement, dst_placement) {
            (Placement::Replicate, Placement::Shard(dim)) if src_axis == dst_axis => {
                sim.local_slice(dst_axis, phase, "wrapper_dtensor", dim, &self.shards)?
            }
            (Placement::Shard(_), Placement::Replicate) if src_axis == dst_axis => {
                let _ = sim.all_gather(src_axis, phase, &self.shards)?;
                let full = self.full_tensor()?;
                vec![full; self.dims.world_size()]
            }
            (Placement::Partial(ReduceOp::Sum), Placement::Replicate) if src_axis == dst_axis => {
                sim.all_reduce_sum(src_axis, phase, &self.shards)?
            }
            (Placement::Partial(ReduceOp::Sum), Placement::Shard(dim)) if src_axis == dst_axis => {
                sim.reduce_scatter_sum(src_axis, dim, phase, &self.shards)?
            }
            _ => {
                return Err(DtError::UnsupportedRedistribution {
                    src: layout_label(&self.layout),
                    dst: layout_label(&dst),
                });
            }
        };

        sim.trace.record(MeshTraceEvent::LayoutTransition {
            phase,
            tensor: "wrapper_dtensor".to_string(),
            src: layout_label(&self.layout),
            dst: layout_label(&dst),
        });

        Ok(Self {
            global_shape: self.global_shape.clone(),
            layout: dst,
            dims: self.dims,
            shards,
        })
    }

    pub fn full_tensor(&self) -> Result<TensorValue, DtError> {
        self.validate_rank_count()?;
        match single_axis_placement(&self.layout)? {
            None | Some((_, Placement::Replicate)) => Ok(self.shards[0].clone()),
            Some((_, Placement::Shard(_))) => {
                let shard_map =
                    ShardMap::new(self.global_shape.clone(), self.layout.clone(), self.dims)?;
                Ok(shard_map.reconstruct_from_rank_shards(&self.shards)?)
            }
            Some((_, Placement::Partial(ReduceOp::Sum))) | Some((_, Placement::Local)) => {
                Err(DtError::UnsupportedRedistribution {
                    src: layout_label(&self.layout),
                    dst: "full_tensor".to_string(),
                })
            }
        }
    }

    fn validate_rank_count(&self) -> Result<(), DtError> {
        let expected = self.dims.world_size();
        let got = self.shards.len();
        if got == expected {
            Ok(())
        } else {
            Err(DtError::RankCountMismatch { expected, got })
        }
    }
}

impl TrainingStepScenario {
    pub fn run_wrapper_dtensor_tracer_bullet(
        &self,
        dims: ParallelDims5D,
        injection_plan: Option<&InjectionPlan>,
    ) -> Result<WrapperDtensorTracerBulletReport, DtError> {
        let reference = self.run_unsharded_reference();
        let mut sim = CollectiveSimulator::new(dims);

        let activation = self.reference_activation_value();
        let tp_replicated = Layout::new(vec![(MeshAxis::Tp, Placement::Replicate)])?;
        let tp_sharded = Layout::new(vec![(MeshAxis::Tp, Placement::Shard(2))])?;
        let activation_dtensor =
            DTensor::from_replicated(activation.clone(), tp_replicated.clone(), dims)?;
        let activation_sharded =
            activation_dtensor.redistribute(tp_sharded, &mut sim, TrainingPhase::Forward)?;
        let activation_tp_shard_shape = activation_sharded.shards[0].shape.clone();
        let activation_replicated =
            activation_sharded.redistribute(tp_replicated, &mut sim, TrainingPhase::Forward)?;
        let activation_full = activation_replicated.full_tensor()?;

        let dp_shard_partial =
            Layout::new(vec![(MeshAxis::DpShard, Placement::Partial(ReduceOp::Sum))])?;
        let dp_sharded = Layout::new(vec![(MeshAxis::DpShard, Placement::Shard(1))])?;
        let mut partial_shards = Vec::with_capacity(dims.world_size());
        for rank in 0..dims.world_size() {
            let offset = rank as f32 * 100.0;
            partial_shards.push(TensorValue::from_vec(
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
            ));
        }
        let mut partial_for_scatter = DTensor {
            global_shape: Shape(vec![2, 4]),
            layout: dp_shard_partial,
            dims,
            shards: partial_shards,
        };

        let mut failures = Vec::new();
        if let Some(plan) = injection_plan {
            for (rank, shard) in partial_for_scatter.shards.iter_mut().enumerate() {
                for event in plan.matching_events(TrainingPhase::Backward, rank) {
                    sim.trace.record(MeshTraceEvent::Injection {
                        phase: event.phase,
                        label: event.label.clone(),
                        rank: event.rank,
                    });
                }
                for mut record in plan.apply_to_shard(TrainingPhase::Backward, rank, shard) {
                    record.tensor = Some("wrapper_partial".to_string());
                    sim.trace.record(MeshTraceEvent::Failure(record.clone()));
                    failures.push(record);
                }
            }
        }

        let partial_sharded =
            partial_for_scatter.redistribute(dp_sharded, &mut sim, TrainingPhase::Backward)?;
        let partial_dp_shard_shape = partial_sharded.shards[0].shape.clone();

        let dp_replicate_partial = Layout::new(vec![(
            MeshAxis::DpReplicate,
            Placement::Partial(ReduceOp::Sum),
        )])?;
        let dp_replicated = Layout::new(vec![(MeshAxis::DpReplicate, Placement::Replicate)])?;
        let partial_for_replicate = DTensor {
            global_shape: Shape(vec![2, 4]),
            layout: dp_replicate_partial,
            dims,
            shards: (0..dims.world_size())
                .map(|rank| TensorValue::from_vec(Shape(vec![2, 4]), vec![rank as f32; 8]))
                .collect(),
        };
        let partial_replicated =
            partial_for_replicate.redistribute(dp_replicated, &mut sim, TrainingPhase::Backward)?;
        let partial_replicated_shape = partial_replicated.shards[0].shape.clone();

        Ok(WrapperDtensorTracerBulletReport {
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

fn require_single_axis(layout: &Layout) -> Result<(MeshAxis, Placement), DtError> {
    match single_axis_placement(layout)? {
        Some(axis_placement) => Ok(axis_placement),
        None => Err(DtError::UnsupportedRedistribution {
            src: layout_label(layout),
            dst: "single-axis distributed layout".to_string(),
        }),
    }
}

fn single_axis_placement(layout: &Layout) -> Result<Option<(MeshAxis, Placement)>, DtError> {
    if layout.axes.len() > 1 {
        return Err(DtError::UnsupportedRedistribution {
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
