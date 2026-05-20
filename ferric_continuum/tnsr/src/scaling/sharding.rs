//! Symbolic sharding cost model for matrix multiplications.
//!
//! Book reference: Ch.3 "Sharded Matrices",
//! https://jax-ml.github.io/scaling-book/sharding/
//!
//! The book describes four cases for distributing Y = X·W across D devices.
//! tnsr is CPU-only and single-threaded — no actual communication happens here.
//! This module is a **local algebra** that estimates per-device FLOPs and
//! all-reduce communication volume for each sharding strategy.
//!
//! Notation (matching the book):
//!   X: M×K input
//!   W: K×N weight
//!   Y: M×N output
//!   D: num_devices (not to be confused with d_model)

/// One of the four canonical sharding cases from Ch.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardingCase {
    /// Shard X along rows (M-dim).  Each device holds M/D rows of X and the
    /// full W.  Output Y is sharded along rows — no communication needed.
    XRowwise,
    /// Shard W along columns (N-dim).  Each device holds M rows of X and N/D
    /// cols of W.  Output Y is sharded along columns — no communication.
    WColwise,
    /// Shard along the inner (K) dimension.  Each device computes a partial
    /// sum; an all-reduce over D devices is required to produce Y.
    InnerReduction,
    /// Output Y is sharded; each device computes its slice and then a
    /// gather is needed to reconstruct the full output on all devices.
    OutputGathered,
}

/// Per-device FLOPs and all-reduce communication volume for one sharded matmul.
#[derive(Debug, Clone)]
pub struct ShardingCost {
    pub case: ShardingCase,
    /// FLOPs per device (forward pass only).
    pub local_fwd_flops: u64,
    /// Bytes moved in the required all-reduce / gather (0 if not needed).
    pub allreduce_bytes: u64,
    /// Number of devices.
    pub num_devices: usize,
}

impl ShardingCost {
    /// Total FLOPs across all devices (same as non-sharded fwd for data parallel).
    pub fn total_fwd_flops(&self) -> u64 {
        self.local_fwd_flops * self.num_devices as u64
    }
}

/// Estimate per-device FLOPs and communication volume for sharding a matmul
/// `Y = X·W` where X is `M×K` and W is `K×N`, across `num_devices` devices.
pub fn shard_matmul(
    m: usize,
    n: usize,
    k: usize,
    num_devices: usize,
    case: ShardingCase,
) -> ShardingCost {
    let d = num_devices as u64;
    let f32_bytes = 4u64;
    let full_fwd: u64 = 2 * m as u64 * k as u64 * n as u64;

    let (local_fwd_flops, allreduce_bytes) = match case {
        ShardingCase::XRowwise => {
            // X is M/D × K per device; W is full K×N.
            // Output Y is M/D × N per device — no all-reduce.
            (full_fwd / d, 0)
        }
        ShardingCase::WColwise => {
            // W is K × N/D per device; X is full M×K.
            // Output Y is M × N/D per device — no all-reduce.
            (full_fwd / d, 0)
        }
        ShardingCase::InnerReduction => {
            // X is M × K/D per device; W is K/D × N per device.
            // Each device computes a partial M×N sum; need all-reduce over D.
            let allreduce = m as u64 * n as u64 * f32_bytes; // M×N f32 elements
            (full_fwd / d, allreduce)
        }
        ShardingCase::OutputGathered => {
            // Like WColwise but need to gather N/D slices into full N on all devices.
            let gather = m as u64 * n as u64 * f32_bytes;
            (full_fwd / d, gather)
        }
    };

    ShardingCost {
        case,
        local_fwd_flops,
        allreduce_bytes,
        num_devices,
    }
}
