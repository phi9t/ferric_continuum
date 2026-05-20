//! Inference memory estimators.
//!
//! Book reference: Ch.7 "Transformer Inference",
//! https://jax-ml.github.io/scaling-book/inference/
//!
//! **Note**: tnsr is a training-only implementation.  The functions in this
//! module are analytical estimates — no KV cache is actually allocated or
//! managed.

/// Estimate the KV cache size for autoregressive inference.
///
/// Formula: `2 × elem_bytes × num_layers × B × T × D`
///
/// The factor of 2 accounts for both K and V tensors per layer.
/// `D` is the full model dimension (not the per-head dimension).
///
/// # Arguments
/// * `num_layers` — number of transformer blocks
/// * `b` — batch size
/// * `t` — maximum sequence length (context window)
/// * `d` — model hidden dimension (d_model)
/// * `elem_bytes` — bytes per element (4 for f32, 2 for bf16/f16)
pub fn kv_cache_bytes(
    num_layers: usize,
    b: usize,
    t: usize,
    d: usize,
    elem_bytes: usize,
) -> u64 {
    2u64 * elem_bytes as u64 * num_layers as u64 * b as u64 * t as u64 * d as u64
}

/// Estimate peak activation memory during a forward pass (no checkpointing).
///
/// Each layer stores roughly `2 × B × T × D` floats of activations for the
/// attention path plus `2 × B × T × D_ff` for the MLP path.
/// This is a rough upper bound; the actual peak depends on execution order.
pub fn peak_activation_bytes(
    num_layers: usize,
    b: usize,
    t: usize,
    d: usize,
    d_ff: usize,
    elem_bytes: usize,
) -> u64 {
    let attn_acts = 2u64 * b as u64 * t as u64 * d as u64;
    let mlp_acts = 2u64 * b as u64 * t as u64 * d_ff as u64;
    (attn_acts + mlp_acts) * elem_bytes as u64 * num_layers as u64
}
