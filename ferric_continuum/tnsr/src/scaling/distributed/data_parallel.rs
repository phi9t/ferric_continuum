//! Data parallelism (DDP) — replicate the model, all-reduce the gradients.
//!
//! Book reference: Ch.5 "Parallelize a Transformer for Training",
//! <https://jax-ml.github.io/scaling-book/training/> (the "data parallelism"
//! section).
//!
//! PyTorch counterpart: [`torch.nn.parallel.DistributedDataParallel`] (DDP).
//!
//! [`torch.nn.parallel.DistributedDataParallel`]: https://pytorch.org/docs/stable/notes/ddp.html
//!
//! # The DDP invariant
//!
//! Every one of the `dp` devices holds a **full replica** of the parameters,
//! gradients, and optimizer state. The only thing that changes with `dp` is:
//!
//! - the global batch is split, so each device sees `1/dp` of the tokens, and
//! - after backward, gradients are **all-reduced** so every replica applies the
//!   same averaged update.
//!
//! Consequently **per-device memory is independent of `dp`** — the property the
//! tests pin. To shard the memory too you need [`super::fsdp`] (ZeRO / FSDP).

use super::super::model_stats::ModelStats;
use super::super::F32_BYTES;
use super::collectives::{collective_cost, Collective};

/// Per-device cost of data-parallel training over `dp` devices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataParallelCost {
    /// Data-parallel degree (`dp`).
    pub dp: usize,
    /// Parameter bytes held on each device — the full replica, DDP invariant.
    pub param_bytes_per_device: u64,
    /// Gradient bytes held on each device — also the full replica.
    pub grad_bytes_per_device: u64,
    /// Optimizer-state bytes per device (e.g. Adam m+v ⇒ multiplier 2).
    pub optimizer_state_bytes_per_device: u64,
    /// Bytes moved per device to all-reduce the gradients each step.
    pub grad_allreduce_bytes_per_device: u64,
}

/// Cost of training `stats`'s block replicated across `dp` devices.
///
/// `optimizer_state_mult` is how many full-size buffers the optimizer keeps per
/// parameter: `0` for plain SGD, `1` for SGD+momentum, `2` for Adam (m and v).
pub fn data_parallel_cost(
    stats: &ModelStats,
    dp: usize,
    optimizer_state_mult: u64,
) -> DataParallelCost {
    let param_bytes = stats.params_total as u64 * F32_BYTES;

    // Gradients mirror the parameters one-for-one; each replica keeps them all.
    let grad_bytes = param_bytes;

    // Gradients are all-reduced (sum then average) across the dp devices.
    let grad_allreduce_bytes_per_device =
        collective_cost(Collective::AllReduce, dp, grad_bytes).comm_bytes_per_device;

    DataParallelCost {
        dp,
        param_bytes_per_device: param_bytes,
        grad_bytes_per_device: grad_bytes,
        optimizer_state_bytes_per_device: param_bytes * optimizer_state_mult,
        grad_allreduce_bytes_per_device,
    }
}
