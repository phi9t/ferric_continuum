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
    UnevenScatter {
        elements: usize,
        parts: usize,
    },
    BroadcastRootOutOfRange {
        root_axis_index: usize,
        axis_size: usize,
    },
    SliceDimOutOfRange {
        dim: usize,
        rank: usize,
    },
    UnevenLocalSlice {
        dim: usize,
        size: usize,
        parts: usize,
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
            tensor: None,
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
            tensor: None,
            bytes: collective_bytes(self.dims.axis_size(axis), shape.bytes_f32() as u64, 1),
            ranks: groups.into_iter().flatten().collect(),
        });
        Ok(out)
    }

    pub fn reduce_scatter_sum(
        &mut self,
        axis: MeshAxis,
        shard_dim: usize,
        phase: TrainingPhase,
        shards: &[TensorValue],
    ) -> Result<Vec<TensorValue>, CollectiveError> {
        validate_rank_values(self.dims, shards)?;
        let shape = validate_same_shape(shards)?;
        let axis_size = self.dims.axis_size(axis);
        if shard_dim >= shape.0.len() {
            return Err(CollectiveError::SliceDimOutOfRange {
                dim: shard_dim,
                rank: 0,
            });
        }
        let full = shape.0[shard_dim];
        if full % axis_size != 0 {
            return Err(CollectiveError::UnevenLocalSlice {
                dim: shard_dim,
                size: full,
                parts: axis_size,
            });
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
            let summed = TensorValue::from_vec(shape.clone(), sum);
            for &rank in group {
                let coord = self
                    .dims
                    .coord(rank)
                    .expect("rank group was built from valid ranks");
                out[rank] = contiguous_slice_along_dim(
                    &summed,
                    shard_dim,
                    coord_axis(coord, axis),
                    axis_size,
                );
            }
        }
        self.trace.record(MeshTraceEvent::Collective {
            phase,
            kind: CollectiveKind::ReduceScatter,
            axis,
            tensor: None,
            bytes: collective_bytes(axis_size, shape.bytes_f32() as u64, 1),
            ranks: groups.into_iter().flatten().collect(),
        });
        Ok(out)
    }

    pub fn broadcast(
        &mut self,
        axis: MeshAxis,
        phase: TrainingPhase,
        root_axis_index: usize,
        shards: &[TensorValue],
    ) -> Result<Vec<TensorValue>, CollectiveError> {
        validate_rank_values(self.dims, shards)?;
        let shape = validate_same_shape(shards)?;
        let axis_size = self.dims.axis_size(axis);
        if root_axis_index >= axis_size {
            return Err(CollectiveError::BroadcastRootOutOfRange {
                root_axis_index,
                axis_size,
            });
        }
        let groups = rank_groups_for_axis(self.dims, axis);
        let mut out = shards.to_vec();
        for group in &groups {
            let root_rank = group[root_axis_index];
            for &rank in group {
                out[rank] = shards[root_rank].clone();
            }
        }
        self.trace.record(MeshTraceEvent::Collective {
            phase,
            kind: CollectiveKind::Broadcast,
            axis,
            tensor: None,
            bytes: collective_bytes(axis_size, shape.bytes_f32() as u64, 1),
            ranks: groups.into_iter().flatten().collect(),
        });
        Ok(out)
    }

    pub fn local_slice(
        &mut self,
        axis: MeshAxis,
        phase: TrainingPhase,
        tensor: &str,
        dim: usize,
        shards: &[TensorValue],
    ) -> Result<Vec<TensorValue>, CollectiveError> {
        validate_rank_values(self.dims, shards)?;
        let axis_size = self.dims.axis_size(axis);
        let mut out = Vec::with_capacity(shards.len());
        for (rank, shard) in shards.iter().enumerate() {
            if dim >= shard.shape.0.len() {
                return Err(CollectiveError::SliceDimOutOfRange { dim, rank });
            }
            let full = shard.shape.0[dim];
            if full % axis_size != 0 {
                return Err(CollectiveError::UnevenLocalSlice {
                    dim,
                    size: full,
                    parts: axis_size,
                });
            }
            let coord = self.dims.coord(rank).expect("rank count was validated");
            out.push(contiguous_slice_along_dim(
                shard,
                dim,
                coord_axis(coord, axis),
                axis_size,
            ));
        }
        self.trace.record(MeshTraceEvent::LayoutTransition {
            phase,
            tensor: tensor.to_string(),
            src: "Replicate".to_string(),
            dst: "Local".to_string(),
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

fn validate_same_shape(shards: &[TensorValue]) -> Result<Shape, CollectiveError> {
    let shape = shards[0].shape.clone();
    for shard in shards {
        if shard.shape != shape {
            return Err(CollectiveError::ShapeMismatch {
                expected: shape,
                got: shard.shape.clone(),
            });
        }
    }
    Ok(shape)
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

fn coord_axis(coord: super::mesh::RankCoord5D, axis: MeshAxis) -> usize {
    match axis {
        MeshAxis::Pp => coord.pp,
        MeshAxis::DpReplicate => coord.dp_replicate,
        MeshAxis::DpShard => coord.dp_shard,
        MeshAxis::Cp => coord.cp,
        MeshAxis::Tp => coord.tp,
    }
}

fn contiguous_slice_along_dim(
    value: &TensorValue,
    dim: usize,
    part: usize,
    parts: usize,
) -> TensorValue {
    let mut local_shape = value.shape.clone();
    let full = value.shape.0[dim];
    let chunk = full / parts;
    local_shape.0[dim] = chunk;
    let outer: usize = value.shape.0[..dim].iter().product();
    let inner: usize = value.shape.0[dim + 1..].iter().product();
    let mut out = Vec::with_capacity(local_shape.numel());
    let data = value.data.as_ref();
    for outer_i in 0..outer {
        let row_base = outer_i * full * inner;
        let start = row_base + part * chunk * inner;
        let end = start + chunk * inner;
        out.extend_from_slice(&data[start..end]);
    }
    TensorValue::from_vec(local_shape, out)
}

fn collective_bytes(axis_size: usize, logical_bytes: u64, multiplier: u64) -> u64 {
    if axis_size <= 1 {
        0
    } else {
        multiplier * (axis_size as u64 - 1) * logical_bytes / axis_size as u64
    }
}
