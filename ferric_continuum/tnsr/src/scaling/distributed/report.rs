//! Aggregate distributed-training report over a 2-D device mesh.
//!
//! Book reference: Ch.5 "Parallelize a Transformer for Training",
//! <https://jax-ml.github.io/scaling-book/training/>.
//!
//! This mirrors [`super::super::report::format_report`] but for the *cross-device*
//! picture: it takes a [`DeviceMesh`] with `"dp"` and `"tp"` axes, computes the
//! per-mechanism costs (DDP grad all-reduce, FSDP sharded memory, tensor-parallel
//! all-reduce), and uses [`super::super::roofline`] to flag whether the whole
//! step is comm-bound or compute-bound on the given hardware.

use crate::transformer::TransformerConfig;

use super::super::model_stats::{model_stats, ModelStats};
use super::super::op_cost::total_train_flops;
use super::super::roofline::{roofline, Bottleneck, HardwareSpec};
use super::data_parallel::{data_parallel_cost, DataParallelCost};
use super::fsdp::{fsdp_memory, FsdpMemory, ZeroStage};
use super::mesh::DeviceMesh;
use super::tensor_parallel::{tensor_parallel_cost, TensorParallelCost};

/// Everything the distributed report aggregates for one config + mesh.
#[derive(Debug, Clone)]
pub struct DistributedReport {
    /// Data-parallel degree read from the mesh's `"dp"` axis (defaults to 1).
    pub dp: usize,
    /// Tensor-parallel degree read from the mesh's `"tp"` axis (defaults to 1).
    pub tp: usize,
    /// Number of transformer blocks in the model.
    pub num_layers: usize,
    /// Per-block parameter breakdown.
    pub stats: ModelStats,
    /// Data-parallel gradient-sync cost.
    pub data_parallel: DataParallelCost,
    /// Sharded-memory picture under the chosen ZeRO stage.
    pub fsdp: FsdpMemory,
    /// Tensor-parallel MLP cost.
    pub tensor_parallel: TensorParallelCost,
    /// Total training FLOPs for the whole model (all layers, fwd + bwd).
    pub total_train_flops: u64,
    /// Total per-device collective bytes across DP + FSDP-extra + TP.
    pub total_comm_bytes_per_device: u64,
    /// Roofline verdict: is the step compute- or comm(memory)-bound on `hw`?
    pub bottleneck: Bottleneck,
}

/// Build the distributed report for `cfg` with `num_layers` blocks laid out on
/// `mesh`, using ZeRO `fsdp_stage` and hardware `hw`.
///
/// The mesh's `"dp"` and `"tp"` axes drive the data- and tensor-parallel
/// degrees; a missing axis defaults to 1 (that mechanism disabled). Adam
/// optimizer state (multiplier 2) is assumed throughout.
pub fn distributed_report(
    cfg: &TransformerConfig,
    num_layers: usize,
    mesh: &DeviceMesh,
    fsdp_stage: ZeroStage,
    hw: &HardwareSpec,
) -> DistributedReport {
    const ADAM_STATE_MULT: u64 = 2;

    let dp = mesh.axis_size("dp").unwrap_or(1);
    let tp = mesh.axis_size("tp").unwrap_or(1);
    let stats = model_stats(cfg);

    let data_parallel = data_parallel_cost(&stats, dp, ADAM_STATE_MULT);
    let fsdp = fsdp_memory(&stats, dp, fsdp_stage, ADAM_STATE_MULT);
    let tensor_parallel = tensor_parallel_cost(cfg, tp);

    let per_layer_flops = total_train_flops(cfg);
    let total_train_flops = per_layer_flops * num_layers as u64;

    // Total per-device communication: DP grad all-reduce + FSDP extra +
    // TP all-reduce (per layer × num_layers for the TP term).
    let total_comm_bytes_per_device = data_parallel.grad_allreduce_bytes_per_device
        + fsdp.extra_comm_bytes_per_device
        + tensor_parallel.allreduce_bytes() * num_layers as u64;

    // Roofline: compare total FLOPs against total bytes moved. If arithmetic
    // intensity clears the ridge point the step is compute-bound; else comm-bound.
    let est = roofline(total_train_flops, total_comm_bytes_per_device, hw);

    DistributedReport {
        dp,
        tp,
        num_layers,
        stats,
        data_parallel,
        fsdp,
        tensor_parallel,
        total_train_flops,
        total_comm_bytes_per_device,
        bottleneck: est.bottleneck,
    }
}

/// Render the distributed report as a multi-line ASCII table (no external deps),
/// in the style of [`super::super::report::format_report`].
pub fn format_distributed_report(r: &DistributedReport) -> String {
    let mut out = String::new();

    out.push_str("=== tnsr Distributed Training Report ===\n");
    out.push_str(&format!(
        "Mesh: dp={} tp={}  (layers={})\n",
        r.dp, r.tp, r.num_layers
    ));
    out.push_str(&format!(
        "Params/block:    {:>12}  (matmul={}, norm={})\n",
        r.stats.params_total, r.stats.params_matmul, r.stats.params_norm
    ));
    out.push_str(&format!(
        "Train FLOPs:     {:>12}  (all layers, fwd+bwd)\n",
        r.total_train_flops
    ));
    out.push('\n');

    out.push_str(&format!(
        "{:<22} {:>16} {:>16}\n",
        "mechanism", "per_dev_bytes", "per_dev_mem"
    ));
    out.push_str(&"-".repeat(56));
    out.push('\n');
    out.push_str(&format!(
        "{:<22} {:>16} {:>16}\n",
        "DDP grad all-reduce",
        r.data_parallel.grad_allreduce_bytes_per_device,
        r.data_parallel.param_bytes_per_device
            + r.data_parallel.grad_bytes_per_device
            + r.data_parallel.optimizer_state_bytes_per_device
    ));
    out.push_str(&format!(
        "{:<22} {:>16} {:>16}\n",
        format!("FSDP {:?}", r.fsdp.stage),
        r.fsdp.extra_comm_bytes_per_device,
        r.fsdp.total_bytes()
    ));
    out.push_str(&format!(
        "{:<22} {:>16} {:>16}\n",
        "Tensor-parallel MLP",
        r.tensor_parallel.allreduce_bytes(),
        "-"
    ));
    out.push_str(&"-".repeat(56));
    out.push('\n');

    out.push_str(&format!(
        "Total comm/device: {:>12}  bytes\n",
        r.total_comm_bytes_per_device
    ));
    out.push_str(&format!("Bottleneck:        {:>12?}\n", r.bottleneck));
    out
}
