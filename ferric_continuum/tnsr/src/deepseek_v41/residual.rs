//! DeepSeek V4.1 hyper-connection residual stream.
//!
//! The text forward path threads a `[B, S, HC, D]` residual buffer between
//! blocks, together with a *pre-mix* carry: the pre-mix produced by layer `N`'s
//! FFN step feeds layer `N+1`'s attention step (and, at the end, the final
//! collapse). Before this type that contract lived in the *caller* — a
//! tuple-return `(Tensor, Vec<f32>)` threaded by hand across `model.rs`, with
//! the "row-major, contiguous, no-grad" layout kept as tribal knowledge.
//!
//! [`ResidualStream`] owns that state:
//! - the flat `[B, S, HC, D]` buffer,
//! - its [`HcShape`],
//! - the carried `pre_mix` (`[B, S, HC]`).
//!
//! It is deliberately a plain value type over `Vec<f32>`, *not* an autograd
//! `Tensor`: the inter-block stream is never differentiated here. Each step
//! runs the upstream `hc_mixes -> hc_pre -> sublayer -> hc_post` sequence and
//! mutates the buffer / pre-mix carry in place, preserving exact op order and
//! `f32` rounding.

use crate::tensor::Tensor;

use super::hc_tensor::{expand_hc, from_flat_bsd, from_flat_bshcd, shape_of, to_flat, HcShape};
use super::hyper::{hc_mixes, hc_post, hc_pre};

/// Per-block hyper-connection weights consumed by one [`ResidualStream::step`].
///
/// These are the six learned vectors upstream splits into the Sinkhorn mixer;
/// grouping them keeps the block-forward call site readable.
pub struct HcStepWeights<'a> {
    pub hc_fn: &'a [f32],
    pub hc_base: &'a [f32],
    pub hc_scale: &'a [f32],
    pub sinkhorn_iters: usize,
    pub eps: f32,
}

/// A hyper-connection residual buffer plus its carried pre-mix.
///
/// Invariant: `buffer.len() == batch*seqlen*hc_mult*dim` and
/// `pre_mix.len() == batch*seqlen*hc_mult` for the owned [`HcShape`].
pub struct ResidualStream {
    buffer: Vec<f32>,
    shape: HcShape,
    pre_mix: Vec<f32>,
}

impl ResidualStream {
    /// Seed a residual stream from a `[B, S, D]` embedding tensor.
    ///
    /// The embedding is broadcast across the hyper-connection axis
    /// ([`expand_hc`]) and the carry starts as the identity pre-mix (lane 0 = 1,
    /// rest 0) — matching the original `try_forward_token_ids` prologue.
    pub fn from_embedding(embed: &Tensor, hc_mult: usize) -> Self {
        let expanded = expand_hc(embed, hc_mult);
        let shape = shape_of(&expanded);
        let pre_mix = identity_pre_mix(shape.batch, shape.seqlen, shape.hc_mult);
        Self {
            buffer: to_flat(&expanded),
            shape,
            pre_mix,
        }
    }

    /// Adopt an already-`[B, S, HC, D]` tensor as the stream buffer.
    ///
    /// Used by the block fixtures, which construct the `[B,S,HC,D]` input
    /// directly and pass an explicit `pre_mix`.
    pub fn from_hc_tensor(x: &Tensor, pre_mix: Vec<f32>) -> Self {
        let shape = shape_of(x);
        assert_eq!(
            pre_mix.len(),
            shape.batch * shape.seqlen * shape.hc_mult,
            "pre_mix length does not match shape"
        );
        Self {
            buffer: to_flat(x),
            shape,
            pre_mix,
        }
    }

    /// Replace the buffer with a `[B, S, HC, D]` tensor (e.g. after an engram
    /// lookup that runs before the block's sublayer steps).
    ///
    /// Shape must be unchanged; the pre-mix carry is untouched.
    pub fn replace_buffer(&mut self, x: &Tensor) {
        let shape = shape_of(x);
        assert_eq!(shape, self.shape, "replace_buffer shape mismatch");
        self.buffer = to_flat(x);
    }

    pub fn shape(&self) -> HcShape {
        self.shape
    }

    /// The carried pre-mix (`[B, S, HC]`), the input to the next step's
    /// `hc_pre`.
    pub fn pre_mix(&self) -> &[f32] {
        &self.pre_mix
    }

    /// Run one hyper-connection sublayer step in place.
    ///
    /// Mirrors the upstream sequence exactly:
    /// `hc_mixes(residual)` -> `hc_pre(residual, carried pre_mix)` ->
    /// `sublayer(pre)` -> `hc_post(sublayer_out, residual, ...)`. The buffer is
    /// overwritten with the `hc_post` result and the pre-mix carry advances to
    /// this step's `pre`.
    ///
    /// `sublayer` receives the `[B, S, D]` pre-mixed tensor and returns the
    /// `[B, S, D]` sublayer output.
    pub fn step(&mut self, weights: HcStepWeights<'_>, sublayer: impl FnOnce(&Tensor) -> Tensor) {
        let shape = self.shape;
        let residual = std::mem::take(&mut self.buffer);
        let mix = hc_mixes(
            &residual,
            weights.hc_fn,
            weights.hc_scale,
            weights.hc_base,
            shape.hc_mult,
            shape.dim,
            weights.sinkhorn_iters,
            weights.eps,
        );
        let pre = from_flat_bsd(shape, hc_pre(&residual, &self.pre_mix, shape));
        let sublayer_out = sublayer(&pre);
        self.buffer = hc_post(
            &to_flat(&sublayer_out),
            &residual,
            &mix.post,
            &mix.comb,
            shape,
        );
        self.pre_mix = mix.pre;
    }

    /// Collapse the `[B, S, HC, D]` buffer to a `[B, S, D]` tensor using the
    /// carried pre-mix. This is the final `hc_pre` before the output norm.
    pub fn collapse(&self) -> Tensor {
        from_flat_bsd(self.shape, hc_pre(&self.buffer, &self.pre_mix, self.shape))
    }

    /// Materialize the current `[B, S, HC, D]` buffer as a no-grad tensor,
    /// without collapsing the hyper-connection axis. Used by the engram lookup,
    /// which reads and rewrites the full residual buffer before the block's
    /// sublayer steps.
    pub fn collapse_hc(&self) -> Tensor {
        from_flat_bshcd(self.shape, self.buffer.clone())
    }
}

/// The identity hyper-connection pre-mix: lane 0 = 1, all other lanes 0.
pub fn identity_pre_mix(batch: usize, seqlen: usize, hc_mult: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * seqlen * hc_mult];
    for token in 0..batch * seqlen {
        out[token * hc_mult] = 1.0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Shape, TensorValue};

    fn tensor(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
        Tensor::from_value_no_grad(TensorValue::from_vec(Shape(shape), data))
    }

    #[test]
    fn identity_pre_mix_sets_lane_zero() {
        assert_eq!(
            identity_pre_mix(1, 2, 3),
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn from_embedding_expands_and_seeds_identity() {
        let embed = tensor(vec![1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let stream = ResidualStream::from_embedding(&embed, 2);
        assert_eq!(stream.shape().hc_mult, 2);
        // token 0 = [1,2] replicated twice, token 1 = [3,4] replicated twice.
        assert_eq!(
            to_flat(&stream.collapse()),
            // collapse with identity pre-mix picks lane 0 only.
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(stream.pre_mix(), &[1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn replace_buffer_keeps_shape_and_pre_mix() {
        let embed = tensor(vec![1, 1, 2], vec![5.0, 6.0]);
        let mut stream = ResidualStream::from_embedding(&embed, 2);
        let carry = stream.pre_mix().to_vec();
        let replacement = tensor(vec![1, 1, 2, 2], vec![7.0, 8.0, 9.0, 10.0]);
        stream.replace_buffer(&replacement);
        assert_eq!(stream.pre_mix(), carry.as_slice());
        assert_eq!(stream.shape().hc_mult, 2);
    }
}
