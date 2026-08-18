use crate::tensor::{Shape, TensorValue};

use super::layout::{Layout, Placement};
use super::mesh::{MeshAxis, MeshError, ParallelDims5D};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShardError {
    Mesh(MeshError),
    MultipleShardAxesUnsupported,
    ShardDimOutOfRange {
        dim: usize,
        rank: usize,
    },
    UnevenShard {
        dim: usize,
        size: usize,
        parts: usize,
    },
    ShapeMismatch {
        expected: Shape,
        got: Shape,
    },
    RankShardCountMismatch {
        expected: usize,
        got: usize,
    },
}

#[derive(Debug, Clone)]
pub struct ShardMap {
    pub global_shape: Shape,
    pub layout: Layout,
    pub dims: ParallelDims5D,
}

impl ShardMap {
    pub fn new(
        global_shape: Shape,
        layout: Layout,
        dims: ParallelDims5D,
    ) -> Result<Self, ShardError> {
        if shard_axes(&layout).len() > 1 {
            return Err(ShardError::MultipleShardAxesUnsupported);
        }
        Ok(Self {
            global_shape,
            layout,
            dims,
        })
    }

    pub fn local_shape_for_rank(&self, rank: usize) -> Result<Shape, ShardError> {
        let mut shape = self.global_shape.clone();
        if let Some((axis, dim)) = single_shard_axis(&self.layout) {
            self.dims.coord(rank).map_err(ShardError::Mesh)?;
            if dim >= shape.0.len() {
                return Err(ShardError::ShardDimOutOfRange { dim, rank });
            }
            let parts = self.dims.axis_size(axis);
            let size = shape.0[dim];
            if size % parts != 0 {
                return Err(ShardError::UnevenShard { dim, size, parts });
            }
            shape.0[dim] = size / parts;
        }
        Ok(shape)
    }

    pub fn shard_tensor_for_rank(
        &self,
        value: &TensorValue,
        rank: usize,
    ) -> Result<TensorValue, ShardError> {
        if value.shape != self.global_shape {
            return Err(ShardError::ShapeMismatch {
                expected: self.global_shape.clone(),
                got: value.shape.clone(),
            });
        }
        if let Some((axis, dim)) = single_shard_axis(&self.layout) {
            let coord = self.dims.coord(rank).map_err(ShardError::Mesh)?;
            self.local_shape_for_rank(rank)?;
            let axis_index = coord_axis(coord, axis);
            contiguous_slice_along_dim(value, dim, axis_index, self.dims.axis_size(axis))
        } else {
            Ok(value.clone())
        }
    }

    pub fn reconstruct_from_rank_shards(
        &self,
        shards: &[TensorValue],
    ) -> Result<TensorValue, ShardError> {
        let world = self.dims.world_size();
        if shards.len() != world {
            return Err(ShardError::RankShardCountMismatch {
                expected: world,
                got: shards.len(),
            });
        }
        for (rank, shard) in shards.iter().enumerate() {
            self.validate_local_shape(&shard.shape, rank)?;
        }
        if let Some((axis, dim)) = single_shard_axis(&self.layout) {
            let parts = self.dims.axis_size(axis);
            let mut selected = Vec::with_capacity(parts);
            for part in 0..parts {
                let rank = (0..world)
                    .find(|rank| {
                        self.dims.coord(*rank).ok().map(|c| coord_axis(c, axis)) == Some(part)
                    })
                    .expect("part rank exists");
                selected.push(shards[rank].clone());
            }
            concat_along_dim(&selected, dim, self.global_shape.clone())
        } else {
            Ok(shards[0].clone())
        }
    }

    fn validate_local_shape(&self, got: &Shape, rank: usize) -> Result<(), ShardError> {
        let expected = self.local_shape_for_rank(rank)?;
        if *got == expected {
            Ok(())
        } else {
            Err(ShardError::ShapeMismatch {
                expected,
                got: got.clone(),
            })
        }
    }
}

fn shard_axes(layout: &Layout) -> Vec<(MeshAxis, usize)> {
    layout
        .axes
        .iter()
        .filter_map(|(axis, placement)| match placement {
            Placement::Shard(dim) => Some((*axis, *dim)),
            _ => None,
        })
        .collect()
}

fn single_shard_axis(layout: &Layout) -> Option<(MeshAxis, usize)> {
    shard_axes(layout).into_iter().next()
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
) -> Result<TensorValue, ShardError> {
    let mut local_shape = value.shape.clone();
    let full = value.shape.0[dim];
    if full % parts != 0 {
        return Err(ShardError::UnevenShard {
            dim,
            size: full,
            parts,
        });
    }
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
    Ok(TensorValue::from_vec(local_shape, out))
}

fn concat_along_dim(
    shards: &[TensorValue],
    dim: usize,
    global_shape: Shape,
) -> Result<TensorValue, ShardError> {
    let parts = shards.len();
    let chunk = shards[0].shape.0[dim];
    let outer: usize = global_shape.0[..dim].iter().product();
    let inner: usize = global_shape.0[dim + 1..].iter().product();
    let mut out = vec![0.0f32; global_shape.numel()];
    for (part, shard) in shards.iter().enumerate() {
        for outer_i in 0..outer {
            let dst_base = outer_i * global_shape.0[dim] * inner + part * chunk * inner;
            let src_base = outer_i * chunk * inner;
            out[dst_base..dst_base + chunk * inner]
                .copy_from_slice(&shard.data[src_base..src_base + chunk * inner]);
        }
    }
    debug_assert_eq!(parts * chunk, global_shape.0[dim]);
    Ok(TensorValue::from_vec(global_shape, out))
}
