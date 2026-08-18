use crate::tensor::{Shape, TensorValue};

use super::mesh::{MeshAxis, ParallelDims5D};
use super::trace::{CollectiveKind, MeshTrace, MeshTraceEvent, TrainingPhase};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollectiveError {
    EmptyInput,
    ShapeMismatch {
        expected: Shape,
        got: Shape,
    },
    RankCountMismatch {
        expected: usize,
        got: usize,
    },
    RankGroupShapeMismatch {
        rank: usize,
        expected: Shape,
        got: Shape,
    },
}

pub struct CollectiveSimulator {
    pub dims: ParallelDims5D,
    pub trace: MeshTrace,
}

impl CollectiveSimulator {
    pub fn new(dims: ParallelDims5D) -> Self {
        Self {
            dims,
            trace: MeshTrace::default(),
        }
    }

    pub fn all_reduce_sum(
        &mut self,
        axis: MeshAxis,
        phase: TrainingPhase,
        shards: &[TensorValue],
    ) -> Result<Vec<TensorValue>, CollectiveError> {
        validate_rank_values(self.dims, shards)?;
        let shape = shards[0].shape.clone();
        for shard in shards {
            if shard.shape != shape {
                return Err(CollectiveError::ShapeMismatch {
                    expected: shape.clone(),
                    got: shard.shape.clone(),
                });
            }
        }
        let groups = rank_groups_for_axis(self.dims, axis);
        let mut out = shards.to_vec();
        for group in &groups {
            let mut sum = vec![0.0f32; shape.numel()];
            for &rank in group {
                for (dst, src) in sum.iter_mut().zip(shards[rank].data.iter()) {
                    *dst += *src;
                }
            }
            for &rank in group {
                out[rank] = TensorValue::from_vec(shape.clone(), sum.clone());
            }
        }
        self.trace.record(MeshTraceEvent::Collective {
            phase,
            kind: CollectiveKind::AllReduce,
            axis,
            bytes: collective_bytes(self.dims.axis_size(axis), shape.bytes_f32() as u64, 2),
            ranks: groups.into_iter().flatten().collect(),
        });
        Ok(out)
    }

    pub fn all_gather(
        &mut self,
        axis: MeshAxis,
        phase: TrainingPhase,
        shards: &[TensorValue],
    ) -> Result<Vec<TensorValue>, CollectiveError> {
        validate_rank_values(self.dims, shards)?;
        let shape = shards[0].shape.clone();
        for shard in shards {
            if shard.shape != shape {
                return Err(CollectiveError::ShapeMismatch {
                    expected: shape.clone(),
                    got: shard.shape.clone(),
                });
            }
        }
        let groups = rank_groups_for_axis(self.dims, axis);
        let mut out = shards.to_vec();
        for group in &groups {
            let mut gathered = Vec::with_capacity(shape.numel() * group.len());
            for &rank in group {
                gathered.extend_from_slice(shards[rank].data.as_ref());
            }
            let out_shape = Shape(vec![group.len(), shape.numel()]);
            for &rank in group {
                out[rank] = TensorValue::from_vec(out_shape.clone(), gathered.clone());
            }
        }
        self.trace.record(MeshTraceEvent::Collective {
            phase,
            kind: CollectiveKind::AllGather,
            axis,
            bytes: collective_bytes(self.dims.axis_size(axis), shape.bytes_f32() as u64, 1),
            ranks: groups.into_iter().flatten().collect(),
        });
        Ok(out)
    }
}

fn validate_rank_values(
    dims: ParallelDims5D,
    shards: &[TensorValue],
) -> Result<(), CollectiveError> {
    if shards.is_empty() {
        return Err(CollectiveError::EmptyInput);
    }
    if shards.len() != dims.world_size() {
        return Err(CollectiveError::RankCountMismatch {
            expected: dims.world_size(),
            got: shards.len(),
        });
    }
    Ok(())
}

fn rank_groups_for_axis(dims: ParallelDims5D, axis: MeshAxis) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut used = vec![false; dims.world_size()];
    for rank in 0..dims.world_size() {
        if used[rank] {
            continue;
        }
        let base = dims.coord(rank).expect("rank in range");
        let mut group = Vec::new();
        for candidate in 0..dims.world_size() {
            let coord = dims.coord(candidate).expect("rank in range");
            let same_other_axes = match axis {
                MeshAxis::Pp => {
                    coord.dp_replicate == base.dp_replicate
                        && coord.dp_shard == base.dp_shard
                        && coord.cp == base.cp
                        && coord.tp == base.tp
                }
                MeshAxis::DpReplicate => {
                    coord.pp == base.pp
                        && coord.dp_shard == base.dp_shard
                        && coord.cp == base.cp
                        && coord.tp == base.tp
                }
                MeshAxis::DpShard => {
                    coord.pp == base.pp
                        && coord.dp_replicate == base.dp_replicate
                        && coord.cp == base.cp
                        && coord.tp == base.tp
                }
                MeshAxis::Cp => {
                    coord.pp == base.pp
                        && coord.dp_replicate == base.dp_replicate
                        && coord.dp_shard == base.dp_shard
                        && coord.tp == base.tp
                }
                MeshAxis::Tp => {
                    coord.pp == base.pp
                        && coord.dp_replicate == base.dp_replicate
                        && coord.dp_shard == base.dp_shard
                        && coord.cp == base.cp
                }
            };
            if same_other_axes {
                used[candidate] = true;
                group.push(candidate);
            }
        }
        groups.push(group);
    }
    groups
}

fn collective_bytes(axis_size: usize, logical_bytes: u64, multiplier: u64) -> u64 {
    if axis_size <= 1 {
        0
    } else {
        multiplier * (axis_size as u64 - 1) * logical_bytes / axis_size as u64
    }
}
