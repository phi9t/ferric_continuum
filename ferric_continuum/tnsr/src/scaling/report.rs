//! Aggregate scaling report for a transformer block.
//!
//! `scale_report` collects `ModelStats` and per-op `OpCost` into one struct
//! and formats it as a human-readable table, mirroring the kind of accounting
//! done in Ch.4 of the JAX scaling book.

use crate::transformer::TransformerConfig;

use super::model_stats::{model_stats, ModelStats};
use super::op_cost::{all_op_costs, OpCost};

/// Complete scaling report for one transformer block.
#[derive(Debug)]
pub struct ScaleReport {
    pub stats: ModelStats,
    pub op_costs: Vec<OpCost>,
    /// Sum of forward FLOPs across all ops.
    pub total_fwd_flops: u64,
    /// Sum of backward FLOPs across all ops.
    pub total_bwd_flops: u64,
    /// Sum of activation bytes produced during the forward pass (no remat).
    pub total_act_bytes: u64,
    /// Total weight bytes (all parameter matrices and vectors).
    pub total_param_bytes: u64,
}

/// Build the scaling report for `cfg`.
pub fn scale_report(cfg: &TransformerConfig) -> ScaleReport {
    let stats = model_stats(cfg);
    let op_costs = all_op_costs(cfg);
    let (total_fwd_flops, total_bwd_flops, total_act_bytes, total_param_bytes) =
        op_costs.iter().fold((0u64, 0u64, 0u64, 0u64), |acc, c| {
            (
                acc.0 + c.fwd_flops,
                acc.1 + c.bwd_flops,
                acc.2 + c.act_bytes,
                acc.3 + c.param_bytes,
            )
        });
    ScaleReport {
        stats,
        op_costs,
        total_fwd_flops,
        total_bwd_flops,
        total_act_bytes,
        total_param_bytes,
    }
}

/// Render the report as a multi-line table string (no external deps).
pub fn format_report(r: &ScaleReport) -> String {
    let mut out = String::new();
    let s = &r.stats;

    out.push_str("=== tnsr Scaling Report ===\n");
    out.push_str(&format!(
        "Config: B={} T={} D={} F={}\n",
        s.batch, s.seq, s.d_model, s.d_ff
    ));
    out.push_str(&format!(
        "Params total:    {:>10}  (matmul={}, norm={})\n",
        s.params_total, s.params_matmul, s.params_norm
    ));
    out.push_str(&format!("  QKVO:          {:>10}\n", s.params_qkvo));
    out.push_str(&format!("  MLP (w1+w2):   {:>10}\n", s.params_mlp));
    out.push_str(&format!("Tokens/batch:    {:>10}\n", s.tokens_per_batch));
    out.push_str(&format!(
        "6*N*T rule:      {:>10}  FLOPs (linear layers only)\n",
        6u64 * s.params_matmul as u64 * s.tokens_per_batch as u64
    ));
    out.push('\n');

    out.push_str(&format!(
        "{:<14} {:>12} {:>12} {:>12} {:>12}\n",
        "op", "fwd_flops", "bwd_flops", "act_bytes", "param_bytes"
    ));
    out.push_str(&"-".repeat(66));
    out.push('\n');

    for c in &r.op_costs {
        out.push_str(&format!(
            "{:<14} {:>12} {:>12} {:>12} {:>12}\n",
            c.name, c.fwd_flops, c.bwd_flops, c.act_bytes, c.param_bytes
        ));
    }

    out.push_str(&"-".repeat(66));
    out.push('\n');
    out.push_str(&format!(
        "{:<14} {:>12} {:>12} {:>12} {:>12}\n",
        "TOTAL", r.total_fwd_flops, r.total_bwd_flops, r.total_act_bytes, r.total_param_bytes
    ));
    out.push_str(&format!(
        "Train FLOPs:     {:>12}  (fwd + bwd)\n",
        r.total_fwd_flops + r.total_bwd_flops
    ));
    out
}
