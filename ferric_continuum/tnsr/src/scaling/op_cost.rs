//! Per-operation FLOPs and byte estimates.
//!
//! All formulas follow Ch.4 of the JAX scaling book.
//! Notation: B=batch, T=seq, D=d_model, F=d_ff.
//!
//! For a matmul Y = X·W with shape (M×K)·(K×N):
//!   fwd  = 2·M·K·N  FLOPs
//!   bwd  = 2×fwd    FLOPs  (∂L/∂X and ∂L/∂W each cost 2·M·K·N)
//!
//! Elementwise ops (GELU, softmax, LayerNorm, residual add) cost O(output_numel)
//! and are tracked separately; they dominate memory bandwidth but not FLOPs
//! at large D.

use crate::transformer::TransformerConfig;

/// FLOPs and memory footprint for one named operation.
#[derive(Debug, Clone)]
pub struct OpCost {
    /// Human-readable operation name.
    pub name: &'static str,
    /// Forward-pass FLOPs (multiply-accumulates count as 2).
    pub fwd_flops: u64,
    /// Backward-pass FLOPs (typically 2× fwd for matmuls).
    pub bwd_flops: u64,
    /// Weight bytes stored (0 for parameter-free ops).
    pub param_bytes: u64,
    /// Activation bytes produced by the forward pass.
    pub act_bytes: u64,
}

impl OpCost {
    /// Training FLOPs = fwd + bwd.
    pub fn train_flops(&self) -> u64 {
        self.fwd_flops + self.bwd_flops
    }
}

/// Return per-op cost estimates for one complete forward pass through `cfg`.
///
/// Ordering matches `TransformerBlock::forward`:
/// ln1 → Q/K/V → attn_scores → causal_mask → softmax → attn_mix →
/// O_proj → residual_1 → ln2 → mlp_up → gelu → mlp_down → residual_2
pub fn all_op_costs(cfg: &TransformerConfig) -> Vec<OpCost> {
    let b = cfg.batch as u64;
    let t = cfg.seq as u64;
    let d = cfg.d_model as u64;
    let f = cfg.d_ff as u64;

    // helpers
    let act = |shape: u64| shape * super::F32_BYTES;
    let lin = |rows: u64, cols: u64| -> (u64, u64) {
        let fwd = 2 * b * t * rows * cols;
        (fwd, 2 * fwd)
    };

    let mut ops = Vec::new();

    // LayerNorm 1: x[B,T,D] → y[B,T,D]; params: gamma[D] + beta[D]
    ops.push(OpCost {
        name: "ln1",
        fwd_flops: 8 * b * t * d,
        bwd_flops: 10 * b * t * d,
        param_bytes: 2 * d * super::F32_BYTES,
        act_bytes: act(b * t * d),
    });

    // Q/K/V projections [B,T,D] @ [D,D] → [B,T,D] — identical FLOPs and bytes
    let (qkv_fwd, qkv_bwd) = lin(d, d);
    let qkv_param = d * d * super::F32_BYTES;
    let qkv_act = act(b * t * d);
    for name in ["q_proj", "k_proj", "v_proj"] {
        ops.push(OpCost {
            name,
            fwd_flops: qkv_fwd,
            bwd_flops: qkv_bwd,
            param_bytes: qkv_param,
            act_bytes: qkv_act,
        });
    }

    // Attention scores: Q[B,T,D] · Kᵀ[B,D,T] → scores[B,T,T]
    // 2·B·T²·D fwd (each of B batches: (T×D)·(D×T) = 2·T²·D)
    let attn_fwd = 2 * b * t * t * d;
    ops.push(OpCost {
        name: "attn_scores",
        fwd_flops: attn_fwd,
        bwd_flops: 2 * attn_fwd,
        param_bytes: 0,
        act_bytes: act(b * t * t),
    });

    // Causal mask: O(B·T²) element writes, no arithmetic FLOPs
    ops.push(OpCost {
        name: "causal_mask",
        fwd_flops: b * t * t,
        bwd_flops: 0,
        param_bytes: 0,
        act_bytes: act(b * t * t),
    });

    // Softmax over last dim [B,T,T]: ~6 ops/element (max, exp, sum, div)
    ops.push(OpCost {
        name: "softmax",
        fwd_flops: 6 * b * t * t,
        bwd_flops: 5 * b * t * t,
        param_bytes: 0,
        act_bytes: act(b * t * t),
    });

    // Attention mix: P[B,T,T] · V[B,T,D] → [B,T,D]
    ops.push(OpCost {
        name: "attn_mix",
        fwd_flops: attn_fwd,
        bwd_flops: 2 * attn_fwd,
        param_bytes: 0,
        act_bytes: act(b * t * d),
    });

    // O projection [B,T,D] @ [D,D] → [B,T,D] — same cost as each of Q/K/V
    ops.push(OpCost {
        name: "o_proj",
        fwd_flops: qkv_fwd,
        bwd_flops: qkv_bwd,
        param_bytes: qkv_param,
        act_bytes: qkv_act,
    });

    // Residual add 1: O(B·T·D), no grad accumulation cost
    ops.push(OpCost {
        name: "residual_1",
        fwd_flops: b * t * d,
        bwd_flops: b * t * d,
        param_bytes: 0,
        act_bytes: act(b * t * d),
    });

    // LayerNorm 2
    ops.push(OpCost {
        name: "ln2",
        fwd_flops: 8 * b * t * d,
        bwd_flops: 10 * b * t * d,
        param_bytes: 2 * d * super::F32_BYTES,
        act_bytes: act(b * t * d),
    });

    // MLP up: [B,T,D] @ [D,F] → [B,T,F]
    let (mlp_up_fwd, mlp_up_bwd) = lin(d, f);
    ops.push(OpCost {
        name: "mlp_up",
        fwd_flops: mlp_up_fwd,
        bwd_flops: mlp_up_bwd,
        param_bytes: d * f * super::F32_BYTES,
        act_bytes: act(b * t * f),
    });

    // GELU: ~6 ops/element (erf approximation)
    ops.push(OpCost {
        name: "gelu",
        fwd_flops: 6 * b * t * f,
        bwd_flops: 6 * b * t * f,
        param_bytes: 0,
        act_bytes: act(b * t * f),
    });

    // MLP down: [B,T,F] @ [F,D] → [B,T,D]
    let (mlp_dn_fwd, mlp_dn_bwd) = lin(f, d);
    ops.push(OpCost {
        name: "mlp_down",
        fwd_flops: mlp_dn_fwd,
        bwd_flops: mlp_dn_bwd,
        param_bytes: f * d * super::F32_BYTES,
        act_bytes: act(b * t * d),
    });

    // Residual add 2
    ops.push(OpCost {
        name: "residual_2",
        fwd_flops: b * t * d,
        bwd_flops: b * t * d,
        param_bytes: 0,
        act_bytes: act(b * t * d),
    });

    ops
}

/// Sum of `fwd_flops` across all ops from `all_op_costs`.
pub fn total_fwd_flops(cfg: &TransformerConfig) -> u64 {
    all_op_costs(cfg).iter().map(|c| c.fwd_flops).sum()
}

/// Sum of `fwd_flops + bwd_flops` across all ops.
pub fn total_train_flops(cfg: &TransformerConfig) -> u64 {
    all_op_costs(cfg).iter().map(|c| c.train_flops()).sum()
}

/// Dense matmul fwd FLOPs only (Q+K+V+O projections + attn_scores + attn_mix
/// + mlp_up + mlp_down). Used to verify the 6·N·T rule.
pub fn dense_matmul_fwd_flops(cfg: &TransformerConfig) -> u64 {
    all_op_costs(cfg)
        .iter()
        .filter(|c| {
            matches!(
                c.name,
                "q_proj"
                    | "k_proj"
                    | "v_proj"
                    | "o_proj"
                    | "attn_scores"
                    | "attn_mix"
                    | "mlp_up"
                    | "mlp_down"
            )
        })
        .map(|c| c.fwd_flops)
        .sum()
}
