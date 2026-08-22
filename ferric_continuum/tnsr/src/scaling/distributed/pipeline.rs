//! Pipeline parallelism — split layers across stages, stream microbatches.
//!
//! Book reference: Ch.5 "Parallelize a Transformer for Training",
//! <https://jax-ml.github.io/scaling-book/training/> (the pipeline-parallelism
//! / bubble section).
//!
//! PyTorch counterpart: [`torch.distributed.pipelining`] (`PipelineSchedule`,
//! GPipe / 1F1B schedules).
//!
//! [`torch.distributed.pipelining`]: https://pytorch.org/docs/stable/distributed.pipelining.html
//!
//! # The pipeline bubble
//!
//! With `P` pipeline stages and `M` microbatches, the pipeline fills and drains
//! over `P−1` steps at each end while only one stage is busy. The fraction of
//! time wasted in this "bubble" (GPipe schedule) is:
//!
//! ```text
//! bubble_fraction = (P − 1) / (M + P − 1)
//! ```
//!
//! It shrinks as `M` grows — more microbatches amortise the fixed fill/drain
//! cost. Between adjacent stages, each microbatch hands off its activation
//! tensor `[B_micro, T, D]`, which is the pipeline's per-boundary comm volume.

use crate::transformer::TransformerConfig;

use super::super::F32_BYTES;

/// Pipeline schedule metrics for `P` stages and `M` microbatches.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineSchedule {
    /// Number of pipeline stages (`P`).
    pub num_stages: usize,
    /// Number of microbatches (`M`).
    pub num_microbatches: usize,
    /// Wasted-time fraction `(P−1)/(M+P−1)` for a GPipe-style schedule.
    pub bubble_fraction: f64,
    /// Bytes handed off between two adjacent stages per microbatch:
    /// one activation tensor `[ceil(B/M), T, D]` in f32.
    pub activation_handoff_bytes: u64,
}

/// Compute the GPipe schedule metrics for `cfg` over `pp` stages and
/// `num_microbatches` microbatches.
///
/// The activation handoff uses the largest microbatch when `cfg.batch` is not
/// evenly divisible by `num_microbatches`; each carries up to
/// `[ceil(B/M), T, D]` elements.
pub fn pipeline_schedule(
    cfg: &TransformerConfig,
    pp: usize,
    num_microbatches: usize,
) -> PipelineSchedule {
    assert!(pp >= 1, "need at least one pipeline stage");
    assert!(num_microbatches >= 1, "need at least one microbatch");

    let p = pp as f64;
    let m = num_microbatches as f64;
    let bubble_fraction = (p - 1.0) / (m + p - 1.0);

    // Activation tensor crossing a stage boundary: [ceil(B/M), T, D] f32 elements.
    let micro_batch = cfg.batch.div_ceil(num_microbatches) as u64;
    let activation_handoff_bytes = micro_batch * cfg.seq as u64 * cfg.d_model as u64 * F32_BYTES;

    PipelineSchedule {
        num_stages: pp,
        num_microbatches,
        bubble_fraction,
        activation_handoff_bytes,
    }
}
