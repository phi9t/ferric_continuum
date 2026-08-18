use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeshAxis {
    Pp,
    DpReplicate,
    DpShard,
    Cp,
    Tp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshError {
    ZeroAxis { axis: MeshAxis },
    WorldSizeMismatch { expected: usize, got: usize },
    RankOutOfRange { rank: usize, world_size: usize },
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeshError::ZeroAxis { axis } => write!(f, "mesh axis {axis:?} must be >= 1"),
            MeshError::WorldSizeMismatch { expected, got } => {
                write!(f, "world size mismatch: expected {expected}, got {got}")
            }
            MeshError::RankOutOfRange { rank, world_size } => {
                write!(f, "rank {rank} out of range for world size {world_size}")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelDims5D {
    pub pp: usize,
    pub dp_replicate: usize,
    pub dp_shard: usize,
    pub cp: usize,
    pub tp: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankCoord5D {
    pub pp: usize,
    pub dp_replicate: usize,
    pub dp_shard: usize,
    pub cp: usize,
    pub tp: usize,
}

impl ParallelDims5D {
    pub fn new(pp: usize, dp_replicate: usize, dp_shard: usize, cp: usize, tp: usize) -> Self {
        for (axis, size) in [
            (MeshAxis::Pp, pp),
            (MeshAxis::DpReplicate, dp_replicate),
            (MeshAxis::DpShard, dp_shard),
            (MeshAxis::Cp, cp),
            (MeshAxis::Tp, tp),
        ] {
            assert!(size >= 1, "mesh axis {axis:?} must be >= 1");
        }
        Self {
            pp,
            dp_replicate,
            dp_shard,
            cp,
            tp,
        }
    }

    pub fn world_size(self) -> usize {
        self.pp * self.dp_replicate * self.dp_shard * self.cp * self.tp
    }

    pub fn validate_world_size(self, world_size: usize) -> Result<(), MeshError> {
        let expected = self.world_size();
        if expected == world_size {
            Ok(())
        } else {
            Err(MeshError::WorldSizeMismatch {
                expected,
                got: world_size,
            })
        }
    }

    pub fn axis_size(self, axis: MeshAxis) -> usize {
        match axis {
            MeshAxis::Pp => self.pp,
            MeshAxis::DpReplicate => self.dp_replicate,
            MeshAxis::DpShard => self.dp_shard,
            MeshAxis::Cp => self.cp,
            MeshAxis::Tp => self.tp,
        }
    }

    pub fn batch_size(self) -> usize {
        self.dp_replicate * self.dp_shard
    }

    pub fn loss_size(self) -> usize {
        self.dp_replicate * self.dp_shard * self.cp
    }

    pub fn fsdp_size(self) -> usize {
        self.dp_shard * self.cp
    }

    pub fn coord(self, rank: usize) -> Result<RankCoord5D, MeshError> {
        let world_size = self.world_size();
        if rank >= world_size {
            return Err(MeshError::RankOutOfRange { rank, world_size });
        }
        let mut r = rank;
        let tp = r % self.tp;
        r /= self.tp;
        let cp = r % self.cp;
        r /= self.cp;
        let dp_shard = r % self.dp_shard;
        r /= self.dp_shard;
        let dp_replicate = r % self.dp_replicate;
        r /= self.dp_replicate;
        let pp = r % self.pp;
        Ok(RankCoord5D {
            pp,
            dp_replicate,
            dp_shard,
            cp,
            tp,
        })
    }
}
