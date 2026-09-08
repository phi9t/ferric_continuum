//! Equal-contiguous sequence ownership for context-parallel attention.
//!
//! This module owns only the mapping between a logical rank's local sequence
//! positions and the complete batch-major sequence. Attention scores,
//! softmax, causal visibility, and gradient equations stay in their operation
//! modules.

use std::fmt;

use crate::tensor::{Shape, TensorValue};

/// A malformed equal-contiguous context-parallel layout request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AttentionLayoutError {
    EmptyShardSet,
    EmptyLocalSequence,
    GlobalSequenceOverflow {
        shard_count: usize,
        local_sequence: usize,
    },
    RankOutOfRange {
        rank: usize,
        shard_count: usize,
    },
    LocalPositionOutOfRange {
        local_position: usize,
        local_sequence: usize,
    },
}

impl fmt::Display for AttentionLayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyShardSet => write!(f, "attention layout requires at least one shard"),
            Self::EmptyLocalSequence => {
                write!(f, "attention layout requires a nonempty local sequence")
            }
            Self::GlobalSequenceOverflow {
                shard_count,
                local_sequence,
            } => write!(
                f,
                "attention layout global sequence overflow: {shard_count} shards * {local_sequence} positions"
            ),
            Self::RankOutOfRange { rank, shard_count } => write!(
                f,
                "attention layout rank {rank} is outside {shard_count} shards"
            ),
            Self::LocalPositionOutOfRange {
                local_position,
                local_sequence,
            } => write!(
                f,
                "attention layout local position {local_position} is outside length {local_sequence}"
            ),
        }
    }
}

impl std::error::Error for AttentionLayoutError {}

/// The complete interval of query positions owned by one logical rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QueryBlock {
    rank: usize,
    start: usize,
    len: usize,
}

impl QueryBlock {
    pub(crate) fn start(self) -> usize {
        self.start
    }

    /// Convert a position within this rank's shard to its global position.
    pub(crate) fn global_position(
        self,
        local_position: usize,
    ) -> Result<usize, AttentionLayoutError> {
        if local_position >= self.len {
            return Err(AttentionLayoutError::LocalPositionOutOfRange {
                local_position,
                local_sequence: self.len,
            });
        }
        Ok(self.start + local_position)
    }
}

/// Equal, nonempty, contiguous sequence ownership across logical ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EqualContiguousAttentionLayout {
    shard_count: usize,
    local_sequence: usize,
    global_sequence: usize,
}

impl EqualContiguousAttentionLayout {
    pub(crate) fn new(
        shard_count: usize,
        local_sequence: usize,
    ) -> Result<Self, AttentionLayoutError> {
        if shard_count == 0 {
            return Err(AttentionLayoutError::EmptyShardSet);
        }
        if local_sequence == 0 {
            return Err(AttentionLayoutError::EmptyLocalSequence);
        }
        let global_sequence = Shape::checked_product(&[shard_count, local_sequence]).ok_or(
            AttentionLayoutError::GlobalSequenceOverflow {
                shard_count,
                local_sequence,
            },
        )?;
        Ok(Self {
            shard_count,
            local_sequence,
            global_sequence,
        })
    }

    pub(crate) fn global_sequence(self) -> usize {
        self.global_sequence
    }

    pub(crate) fn query_block(self, rank: usize) -> Result<QueryBlock, AttentionLayoutError> {
        if rank >= self.shard_count {
            return Err(AttentionLayoutError::RankOutOfRange {
                rank,
                shard_count: self.shard_count,
            });
        }
        Ok(QueryBlock {
            rank,
            start: rank * self.local_sequence,
            len: self.local_sequence,
        })
    }

    /// Gather `[B,local_sequence,H,Dh]` shards into batch-major
    /// `[B,global_sequence,H,Dh]` storage.
    pub(crate) fn gather_bshd(self, shards: &[TensorValue]) -> TensorValue {
        assert_eq!(shards.len(), self.shard_count);
        let local_shape = &shards[0].shape.0;
        assert_eq!(local_shape.len(), 4);
        let (batch, local_sequence, heads, head_dim) = (
            local_shape[0],
            local_shape[1],
            local_shape[2],
            local_shape[3],
        );
        assert_eq!(local_sequence, self.local_sequence);
        for shard in shards {
            assert_eq!(shard.shape.0.as_slice(), local_shape.as_slice());
        }
        let full_shape = Shape(vec![batch, self.global_sequence, heads, head_dim]);
        let mut full = vec![0.0f32; full_shape.numel()];

        for (rank, shard) in shards.iter().enumerate() {
            let block = self.query_block(rank).expect("validated attention rank");
            let source = shard.data.as_ref();
            for batch_index in 0..batch {
                for local_position in 0..self.local_sequence {
                    let global_position = block
                        .global_position(local_position)
                        .expect("validated local attention position");
                    for head in 0..heads {
                        let source_start =
                            ((batch_index * self.local_sequence + local_position) * heads + head)
                                * head_dim;
                        let target_start =
                            ((batch_index * self.global_sequence + global_position) * heads + head)
                                * head_dim;
                        full[target_start..target_start + head_dim]
                            .copy_from_slice(&source[source_start..source_start + head_dim]);
                    }
                }
            }
        }
        TensorValue::from_vec(full_shape, full)
    }

    /// Scatter batch-major `[B,global_sequence,H,Dh]` storage back to its
    /// equal `[B,local_sequence,H,Dh]` owners.
    pub(crate) fn scatter_bshd(self, full: &TensorValue) -> Vec<TensorValue> {
        let full_shape = &full.shape.0;
        assert_eq!(full_shape.len(), 4);
        let (batch, global_sequence, heads, head_dim) =
            (full_shape[0], full_shape[1], full_shape[2], full_shape[3]);
        assert_eq!(global_sequence, self.global_sequence);
        assert_eq!(full.data.len(), full.shape.numel());

        (0..self.shard_count)
            .map(|rank| {
                let block = self.query_block(rank).expect("validated attention rank");
                let shard_shape = Shape(vec![batch, self.local_sequence, heads, head_dim]);
                let mut shard = vec![0.0f32; shard_shape.numel()];
                for batch_index in 0..batch {
                    for local_position in 0..self.local_sequence {
                        let global_position = block
                            .global_position(local_position)
                            .expect("validated local attention position");
                        for head in 0..heads {
                            let source_start =
                                ((batch_index * self.global_sequence + global_position) * heads
                                    + head)
                                    * head_dim;
                            let target_start =
                                ((batch_index * self.local_sequence + local_position) * heads
                                    + head)
                                    * head_dim;
                            shard[target_start..target_start + head_dim]
                                .copy_from_slice(&full.data[source_start..source_start + head_dim]);
                        }
                    }
                }
                TensorValue::from_vec(shard_shape, shard)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_query_block_has_an_explicit_global_interval() {
        let layout = EqualContiguousAttentionLayout::new(4, 2).unwrap();

        assert_eq!(layout.shard_count, 4);
        assert_eq!(layout.local_sequence, 2);
        assert_eq!(layout.global_sequence(), 8);
        for rank in 0..4 {
            let block = layout.query_block(rank).unwrap();
            assert_eq!(block.rank, rank);
            assert_eq!(block.start(), rank * 2);
            assert_eq!(block.len, 2);
            assert_eq!(block.start + block.len, (rank + 1) * 2);
            assert_eq!(block.global_position(0).unwrap(), rank * 2);
            assert_eq!(block.global_position(1).unwrap(), rank * 2 + 1);
        }
    }

    #[test]
    fn invalid_geometry_and_indices_are_reported_before_indexing() {
        assert_eq!(
            EqualContiguousAttentionLayout::new(0, 2),
            Err(AttentionLayoutError::EmptyShardSet)
        );
        assert_eq!(
            EqualContiguousAttentionLayout::new(2, 0),
            Err(AttentionLayoutError::EmptyLocalSequence)
        );
        assert_eq!(
            EqualContiguousAttentionLayout::new(2, usize::MAX),
            Err(AttentionLayoutError::GlobalSequenceOverflow {
                shard_count: 2,
                local_sequence: usize::MAX,
            })
        );

        let layout = EqualContiguousAttentionLayout::new(2, 3).unwrap();
        assert_eq!(
            layout.query_block(2),
            Err(AttentionLayoutError::RankOutOfRange {
                rank: 2,
                shard_count: 2,
            })
        );
        assert_eq!(
            layout.query_block(0).unwrap().global_position(3),
            Err(AttentionLayoutError::LocalPositionOutOfRange {
                local_position: 3,
                local_sequence: 3,
            })
        );
    }

    #[test]
    fn gather_and_scatter_round_trip_asymmetric_batch_major_values() {
        let layout = EqualContiguousAttentionLayout::new(2, 2).unwrap();
        let first = TensorValue::from_vec(Shape(vec![2, 2, 1, 1]), vec![0.0, 1.0, 10.0, 11.0]);
        let second = TensorValue::from_vec(Shape(vec![2, 2, 1, 1]), vec![2.0, 3.0, 12.0, 13.0]);

        let full = layout.gather_bshd(&[first.clone(), second.clone()]);
        assert_eq!(full.shape.0, vec![2, 4, 1, 1]);
        assert_eq!(
            full.data.as_ref(),
            &[0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]
        );
        let scattered = layout.scatter_bshd(&full);
        assert_eq!(scattered.len(), 2);
        assert_eq!(scattered[0].shape, first.shape);
        assert_eq!(scattered[0].data, first.data);
        assert_eq!(scattered[1].shape, second.shape);
        assert_eq!(scattered[1].data, second.data);
    }
}
