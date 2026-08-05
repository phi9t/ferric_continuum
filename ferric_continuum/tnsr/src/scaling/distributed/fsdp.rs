//! FSDP / ZeRO — shard the optimizer state, gradients, and parameters.
//!
//! Book reference: Ch.5 "Parallelize a Transformer for Training",
//! <https://jax-ml.github.io/scaling-book/training/> (the FSDP / ZeRO section).
//!
//! PyTorch counterpart: [`torch.distributed.fsdp.FullyShardedDataParallel`]
//! (FSDP), which implements ZeRO-3. The intermediate ZeRO stages match
//! DeepSpeed's `zero_optimization.stage`.
//!
//! [`torch.distributed.fsdp.FullyShardedDataParallel`]: https://pytorch.org/docs/stable/fsdp.html
//!
//! # ZeRO ladder (each stage shards strictly more than the last)
//!
//! Starting from DDP (full replicas everywhere), ZeRO progressively partitions
//! the three big memory consumers across the `D` data-parallel devices:
//!
//! | Stage | Optimizer state | Gradients | Parameters | PyTorch/DeepSpeed |
//! |-------|-----------------|-----------|------------|-------------------|
//! | 1     | `1/D`           | full      | full       | ZeRO-1            |
//! | 2     | `1/D`           | `1/D`     | full       | ZeRO-2            |
//! | 3     | `1/D`           | `1/D`     | `1/D`      | ZeRO-3 == FSDP    |
//!
//! Sharding trades memory for communication: Stage 3 must **all-gather**
//! parameters before each matmul and **reduce-scatter** gradients after
//! backward, whereas DDP only pays one gradient all-reduce (see
//! [`super::data_parallel`]). Per-device memory is therefore monotonically
//! non-increasing Stage1 ≥ Stage2 ≥ Stage3 — the ordering the tests pin.

use super::super::model_stats::ModelStats;
use super::super::F32_BYTES;
use super::collectives::{collective_cost, Collective};

/// The three ZeRO partitioning stages. `Stage3` is equivalent to FSDP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroStage {
    /// ZeRO-1: shard optimizer state only.
    Stage1,
    /// ZeRO-2: shard optimizer state + gradients.
    Stage2,
    /// ZeRO-3 / FSDP: shard optimizer state + gradients + parameters.
    Stage3,
}

/// Per-device memory and extra communication for one ZeRO stage over `D` shards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsdpMemory {
    /// Which ZeRO stage this describes.
    pub stage: ZeroStage,
    /// Shard count (`D`), the data-parallel degree the memory is split over.
    pub num_shards: usize,
    /// Parameter bytes per device (`full` or `full/D` depending on stage).
    pub param_bytes: u64,
    /// Gradient bytes per device.
    pub grad_bytes: u64,
    /// Optimizer-state bytes per device (always sharded from Stage 1 up).
    pub optimizer_state_bytes: u64,
    /// Extra collective bytes per device vs DDP: param all-gather (Stage 3) +
    /// gradient reduce-scatter (Stage 2/3). Stage 1 adds nothing here.
    pub extra_comm_bytes_per_device: u64,
}

impl FsdpMemory {
    /// Total sharded footprint per device = params + grads + optimizer state.
    pub fn total_bytes(&self) -> u64 {
        self.param_bytes + self.grad_bytes + self.optimizer_state_bytes
    }
}

/// Per-device memory + extra comm for `stats` under `stage`, sharded `num_shards` ways.
///
/// `optimizer_state_mult` is the per-parameter optimizer buffer count (2 for
/// Adam's m+v); it scales the optimizer-state term only.
pub fn fsdp_memory(
    stats: &ModelStats,
    num_shards: usize,
    stage: ZeroStage,
    optimizer_state_mult: u64,
) -> FsdpMemory {
    let d = num_shards.max(1) as u64;
    let full_params = stats.params_total as u64 * F32_BYTES;
    let full_grads = full_params; // one gradient per parameter
    let full_opt = full_params * optimizer_state_mult;

    // Optimizer state is sharded at every ZeRO stage.
    let optimizer_state_bytes = full_opt / d;

    // Gradients shard from Stage 2 up; parameters only at Stage 3.
    let (param_bytes, grad_bytes) = match stage {
        ZeroStage::Stage1 => (full_params, full_grads),
        ZeroStage::Stage2 => (full_params, full_grads / d),
        ZeroStage::Stage3 => (full_params / d, full_grads / d),
    };

    // Extra collectives vs DDP's single gradient all-reduce:
    //   Stage 2/3 reduce-scatter gradients instead of all-reducing them;
    //   Stage 3 additionally all-gathers parameters before each matmul.
    let grad_rs = match stage {
        ZeroStage::Stage1 => 0,
        ZeroStage::Stage2 | ZeroStage::Stage3 => {
            collective_cost(Collective::ReduceScatter, num_shards, full_grads).comm_bytes_per_device
        }
    };
    let param_ag = match stage {
        ZeroStage::Stage1 | ZeroStage::Stage2 => 0,
        ZeroStage::Stage3 => {
            collective_cost(Collective::AllGather, num_shards, full_params).comm_bytes_per_device
        }
    };

    FsdpMemory {
        stage,
        num_shards,
        param_bytes,
        grad_bytes,
        optimizer_state_bytes,
        extra_comm_bytes_per_device: grad_rs + param_ag,
    }
}
