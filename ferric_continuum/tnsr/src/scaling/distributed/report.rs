//! Aggregate distributed-training report over a 2-D device mesh.
//!
//! Book reference: Ch.5 "Parallelize a Transformer for Training",
//! <https://jax-ml.github.io/scaling-book/training/>.
//!
//! This mirrors [`super::super::report::format_report`] but for the *cross-device*
//! picture: it takes a [`DeviceMesh`] with named axes, computes the
//! per-mechanism costs (stage-aware DP/FSDP gradient sync, FSDP sharded memory,
//! tensor-parallel all-reduce, pipeline handoff), and uses
//! [`super::super::roofline`] to flag whether the whole step is comm-bound or
//! compute-bound on the given hardware.

use crate::transformer::TransformerConfig;

use super::super::model_stats::{model_stats, ModelStats};
use super::super::op_cost::total_train_flops;
use super::super::roofline::{roofline, Bottleneck, HardwareSpec};
use super::data_parallel::{data_parallel_cost, DataParallelCost};
use super::fsdp::{fsdp_memory, FsdpMemory, ZeroStage};
use super::mesh::DeviceMesh;
use super::pipeline::{pipeline_schedule, PipelineSchedule};
use super::tensor_parallel::{tensor_parallel_cost, TensorParallelCost};

const DEFAULT_MICROBATCHES_PER_PIPELINE_STAGE: usize = 1;

/// A named collective implied by the selected parallelism axes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolicCollective {
    /// Human-facing name of the logical communication event.
    pub name: &'static str,
    /// Number of logical devices participating along that axis.
    pub participants: usize,
    /// Why this communication appears in the training step.
    pub reason: &'static str,
}

/// Everything the distributed report aggregates for one config + mesh.
#[derive(Debug, Clone)]
pub struct DistributedReport {
    /// Data-parallel degree read from the mesh's `"dp"` axis (defaults to 1).
    pub dp: usize,
    /// Tensor-parallel degree read from the mesh's `"tp"` axis (defaults to 1).
    pub tp: usize,
    /// Pipeline-parallel degree read from the mesh's `"pp"` axis (defaults to 1).
    pub pp: usize,
    /// Sequence/context-parallel degree from `"sp"` or `"cp"` (defaults to 1).
    pub sequence_parallel: usize,
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
    /// Pipeline schedule summary.
    pub pipeline: PipelineSchedule,
    /// Number of schedule microbatches (`M`) chosen for the pipeline model.
    pub num_pipeline_microbatches: usize,
    /// Largest batch chunk in a schedule microbatch.
    pub pipeline_microbatch_size: usize,
    /// Total logical activation handoff bytes across all adjacent pipeline
    /// boundaries and schedule microbatches.
    pub pipeline_step_handoff_bytes: u64,
    /// Activation sharding multiplier from TP plus sequence/context parallelism.
    pub activation_shard_factor: usize,
    /// Symbolic collectives implied by the active axes and ZeRO stage.
    pub symbolic_collectives: Vec<SymbolicCollective>,
    /// Total training FLOPs for the whole model (all layers, fwd + bwd).
    pub total_train_flops: u64,
    /// Total per-device collective bytes across the stage-aware DP/FSDP path,
    /// TP, and PP handoff.
    pub total_comm_bytes_per_device: u64,
    /// Roofline verdict: is the step compute- or comm(memory)-bound on `hw`?
    pub bottleneck: Bottleneck,
}

/// Build the distributed report for `cfg` with `num_layers` blocks laid out on
/// `mesh`, using ZeRO `fsdp_stage` and hardware `hw`.
///
/// The mesh's `"dp"`, `"tp"`, `"pp"`, and `"sp"`/`"cp"` axes drive the
/// data-, tensor-, pipeline-, and sequence/context-parallel degrees; a missing
/// axis defaults to 1 (that mechanism disabled). Adam optimizer state
/// (multiplier 2) is assumed throughout.
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
    let pp = mesh.axis_size("pp").unwrap_or(1);
    let sequence_parallel = mesh
        .axis_size("sp")
        .or_else(|| mesh.axis_size("cp"))
        .unwrap_or(1);
    let stats = model_stats(cfg);

    let data_parallel = data_parallel_cost(&stats, dp, ADAM_STATE_MULT);
    let fsdp = fsdp_memory(&stats, dp, fsdp_stage, ADAM_STATE_MULT);
    let tensor_parallel = tensor_parallel_cost(cfg, tp);
    let num_pipeline_microbatches = pp * DEFAULT_MICROBATCHES_PER_PIPELINE_STAGE;
    let pipeline_microbatch_size = cfg.batch.div_ceil(num_pipeline_microbatches);
    let pipeline = pipeline_schedule(cfg, pp, num_pipeline_microbatches);
    let pipeline_step_handoff_bytes = pipeline.activation_handoff_bytes
        * num_pipeline_microbatches as u64
        * pp.saturating_sub(1) as u64;
    let activation_shard_factor = tp * sequence_parallel;
    let symbolic_collectives = symbolic_collectives(dp, tp, pp, sequence_parallel, fsdp_stage);

    let per_layer_flops = total_train_flops(cfg);
    let total_train_flops = per_layer_flops * num_layers as u64;

    // Total per-device communication: ZeRO-1 keeps DDP's gradient all-reduce,
    // while ZeRO-2/3 replace it with FSDP reduce-scatter plus any parameter
    // all-gather. Add TP all-reduce (per layer × num_layers) and logical PP
    // activation handoff across schedule microbatches and adjacent stage
    // boundaries. The symbolic context-parallel exchange is listed below but
    // left out of the byte total until the attention module has a byte model.
    let data_shard_comm_bytes_per_device = match fsdp_stage {
        ZeroStage::Stage1 => data_parallel.grad_allreduce_bytes_per_device,
        ZeroStage::Stage2 | ZeroStage::Stage3 => fsdp.extra_comm_bytes_per_device,
    };
    let total_comm_bytes_per_device = data_shard_comm_bytes_per_device
        + tensor_parallel.allreduce_bytes() * num_layers as u64
        + pipeline_step_handoff_bytes;

    // Roofline: compare total FLOPs against total bytes moved. If arithmetic
    // intensity clears the ridge point the step is compute-bound; else comm-bound.
    let est = roofline(total_train_flops, total_comm_bytes_per_device, hw);

    DistributedReport {
        dp,
        tp,
        pp,
        sequence_parallel,
        num_layers,
        stats,
        data_parallel,
        fsdp,
        tensor_parallel,
        pipeline,
        num_pipeline_microbatches,
        pipeline_microbatch_size,
        pipeline_step_handoff_bytes,
        activation_shard_factor,
        symbolic_collectives,
        total_train_flops,
        total_comm_bytes_per_device,
        bottleneck: est.bottleneck,
    }
}

fn symbolic_collectives(
    dp: usize,
    tp: usize,
    pp: usize,
    sequence_parallel: usize,
    fsdp_stage: ZeroStage,
) -> Vec<SymbolicCollective> {
    let mut collectives = Vec::new();

    if dp > 1 && matches!(fsdp_stage, ZeroStage::Stage1) {
        collectives.push(SymbolicCollective {
            name: "DDP gradient all-reduce",
            participants: dp,
            reason: "ZeRO-1 keeps replicated gradients synchronized",
        });
    }

    if matches!(fsdp_stage, ZeroStage::Stage3) && dp > 1 {
        collectives.push(SymbolicCollective {
            name: "FSDP parameter all-gather",
            participants: dp,
            reason: "ZeRO-3 gathers parameter shards before compute",
        });
    }
    if matches!(fsdp_stage, ZeroStage::Stage2 | ZeroStage::Stage3) && dp > 1 {
        collectives.push(SymbolicCollective {
            name: "FSDP gradient reduce-scatter",
            participants: dp,
            reason: "ZeRO-2/3 partitions reduced gradients across data shards",
        });
    }

    if tp > 1 {
        collectives.push(SymbolicCollective {
            name: "TP row-parallel all-reduce",
            participants: tp,
            reason: "row-sharded matmul partial sums need reduction",
        });
    }

    if pp > 1 {
        collectives.push(SymbolicCollective {
            name: "PP activation send/recv",
            participants: pp,
            reason: "activation and gradient tensors cross stage boundaries",
        });
    }

    if sequence_parallel > 1 {
        collectives.push(SymbolicCollective {
            name: "context-parallel K/V exchange",
            participants: sequence_parallel,
            reason: "attention over sequence shards must exchange context",
        });
    }

    collectives
}

/// Render the distributed report as a multi-line ASCII table (no external deps),
/// in the style of [`super::super::report::format_report`].
pub fn format_distributed_report(r: &DistributedReport) -> String {
    let mut out = String::new();

    out.push_str("=== tnsr Distributed Training Report ===\n");
    out.push_str(&format!(
        "Mesh: dp={} tp={} pp={} sp={}  (layers={})\n",
        r.dp, r.tp, r.pp, r.sequence_parallel, r.num_layers
    ));
    out.push_str(&format!(
        "Params/block:    {:>12}  (matmul={}, norm={})\n",
        r.stats.params_total, r.stats.params_matmul, r.stats.params_norm
    ));
    out.push_str(&format!(
        "Train FLOPs:     {:>12}  (all layers, fwd+bwd)\n",
        r.total_train_flops
    ));
    out.push_str(&format!(
        "Pipeline microbatches:{:>8}  (M)\n",
        r.num_pipeline_microbatches
    ));
    out.push_str(&format!(
        "Pipeline microbatch size:{:>5}  (ceil(batch/M))\n",
        r.pipeline_microbatch_size
    ));
    out.push_str(&format!(
        "Activation shard factor:{:>6}  (tp*sp)\n",
        r.activation_shard_factor
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
    out.push_str(&format!(
        "{:<22} {:>16} {:>16}\n",
        "Pipeline handoff", r.pipeline_step_handoff_bytes, "-"
    ));
    out.push_str(&"-".repeat(56));
    out.push('\n');

    out.push_str(&format!(
        "Total comm/device: {:>12}  bytes\n",
        r.total_comm_bytes_per_device
    ));
    out.push_str(&format!("Bottleneck:        {:>12?}\n", r.bottleneck));
    out.push_str("Symbolic collectives:\n");
    for collective in &r.symbolic_collectives {
        out.push_str(&format!(
            "- {} participants={} reason={}\n",
            collective.name, collective.participants, collective.reason
        ));
    }
    out
}
