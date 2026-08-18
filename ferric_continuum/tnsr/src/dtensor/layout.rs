use std::collections::HashSet;

use super::mesh::MeshAxis;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    Replicate,
    Shard(usize),
    Partial(ReduceOp),
    Local,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    DuplicateAxis(MeshAxis),
    LocalMixedWithDistributed,
    DuplicateShardDim(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub axes: Vec<(MeshAxis, Placement)>,
}

impl Layout {
    pub fn new(axes: Vec<(MeshAxis, Placement)>) -> Result<Self, LayoutError> {
        let mut seen_axes = HashSet::new();
        let mut seen_shard_dims = HashSet::new();
        let mut has_local = false;
        let mut has_distributed = false;
        for (axis, placement) in &axes {
            if !seen_axes.insert(*axis) {
                return Err(LayoutError::DuplicateAxis(*axis));
            }
            match placement {
                Placement::Local => has_local = true,
                Placement::Shard(dim) => {
                    has_distributed = true;
                    if !seen_shard_dims.insert(*dim) {
                        return Err(LayoutError::DuplicateShardDim(*dim));
                    }
                }
                Placement::Replicate | Placement::Partial(ReduceOp::Sum) => {
                    has_distributed = true;
                }
            }
        }
        if has_local && has_distributed {
            return Err(LayoutError::LocalMixedWithDistributed);
        }
        Ok(Self { axes })
    }

    pub fn placement(&self, axis: MeshAxis) -> Option<Placement> {
        self.axes
            .iter()
            .find_map(|(a, p)| if *a == axis { Some(*p) } else { None })
    }
}
