//! Parameter counting for a single transformer block.
//!
//! Book reference: Ch.4 "Transformer Accounting" —
//! https://jax-ml.github.io/scaling-book/transformers/
//!
//! The `6 × params × tokens` training-FLOPs rule applies to `params_matmul`
//! (Q+K+V+O projections plus the two MLP weight matrices).  LayerNorm
//! parameters (`params_norm`) are excluded because they are elementwise
//! scalars, not matrix multiplications.

use crate::transformer::TransformerConfig;

/// Parameter breakdown for one transformer block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelStats {
    /// All trainable parameters: matmul + norm.
    pub params_total: usize,
    /// Q + K + V + O projection weights: 4 × D²
    pub params_qkvo: usize,
    /// MLP weight matrices w1 + w2: 2 × D×F  (2-matrix GELU variant)
    pub params_mlp: usize,
    /// LayerNorm gamma + beta for both norm layers: 4 × D
    pub params_norm: usize,
    /// params_qkvo + params_mlp — the "N" in the 6·N·T training FLOPs rule
    pub params_matmul: usize,
    /// B × T — tokens processed per forward pass
    pub tokens_per_batch: usize,
    /// Batch size (B)
    pub batch: usize,
    /// Sequence length (T)
    pub seq: usize,
    /// Model hidden dimension (D)
    pub d_model: usize,
    /// FFN hidden dimension (F)
    pub d_ff: usize,
}

/// Compute parameter counts for one block described by `cfg`.
///
/// Stores `batch` and `seq` so callers don't need to retain the config.
///
/// **Note on MLP variant:** `TransformerBlock` uses the classic 2-matrix GELU
/// FFN (`w1: [D,F]`, `w2: [F,D]`), giving `params_mlp = 2·D·F`.  The book's
/// reference model uses a gated 3-matrix FFN (`3·D·F` params); that variant
/// is available as `ops/activations.rs::swiglu` but is not wired into
/// `TransformerBlock`.
pub fn model_stats(cfg: &TransformerConfig) -> ModelStats {
    let d = cfg.d_model;
    let f = cfg.d_ff;

    let params_qkvo = 4 * d * d;
    let params_mlp = 2 * d * f;
    let params_norm = 4 * d; // ln1_gamma, ln1_beta, ln2_gamma, ln2_beta
    let params_matmul = params_qkvo + params_mlp;
    let params_total = params_matmul + params_norm;
    let tokens_per_batch = cfg.batch * cfg.seq;

    ModelStats {
        params_total,
        params_qkvo,
        params_mlp,
        params_norm,
        params_matmul,
        tokens_per_batch,
        batch: cfg.batch,
        seq: cfg.seq,
        d_model: d,
        d_ff: f,
    }
}
