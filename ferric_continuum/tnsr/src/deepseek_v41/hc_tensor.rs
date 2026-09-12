//! DeepSeek V4.1 hyper-connection tensor <-> flat conversion seam.
//!
//! The DeepSeek text forward path threads a residual stream between blocks as a
//! flat `Vec<f32>` in row-major `[B, S, (HC,) D]` layout with no autograd tape.
//! Before this module that contract lived in the *caller*: every block, layer,
//! and op independently did `x.inner.borrow().value.data.as_ref().clone()` and
//! re-wrapped the result with a hand-specified `Shape`, so the "row-major,
//! contiguous, no-grad" invariant was tribal knowledge and untested.
//!
//! This is the single seam for that conversion.  It exposes:
//! - [`to_flat`] — the one borrow-and-clone site,
//! - [`shape_of`] — read a 4D `[B,S,HC,D]` [`HcShape`],
//! - [`from_flat_bsd`] / [`from_flat_bshcd`] — wrap a flat buffer back into a
//!   `[B,S,D]` / `[B,S,HC,D]` no-grad tensor, asserting the numel matches,
//! - [`expand_hc`] — replicate a `[B,S,D]` tensor across the HC axis.
//!
//! [`HcShape`] lives here because it is fundamentally a layout descriptor, not
//! hyper-connection math; `hyper.rs` re-exports it for existing importers.

use crate::tensor::{Shape, Tensor, TensorValue};

/// Row-major layout of a hyper-connection residual buffer, `[B, S, HC, D]`.
///
/// A `[B, S, D]` view simply drops the `hc_mult` axis; the same struct labels
/// both by convention (see [`from_flat_bsd`] vs [`from_flat_bshcd`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HcShape {
    pub batch: usize,
    pub seqlen: usize,
    pub hc_mult: usize,
    pub dim: usize,
}

/// Borrow a tensor's value and clone its flat, row-major `Vec<f32>`.
///
/// This is the single place the DeepSeek forward path reaches into a `Tensor`'s
/// storage; every prior ad-hoc `inner.borrow().value.data.as_ref().clone()` is
/// folded into this call.
pub fn to_flat(x: &Tensor) -> Vec<f32> {
    x.inner.borrow().value.data.as_ref().clone()
}

/// Read the 4D `[B, S, HC, D]` shape of a hyper-connection residual tensor.
///
/// Panics if the tensor is not 4D — the residual stream between blocks is always
/// `[B, S, HC, D]`.
pub fn shape_of(x: &Tensor) -> HcShape {
    let shape = x.shape().0;
    assert_eq!(shape.len(), 4, "hc_tensor::shape_of expects [B,S,HC,D]");
    HcShape {
        batch: shape[0],
        seqlen: shape[1],
        hc_mult: shape[2],
        dim: shape[3],
    }
}

/// Wrap a flat, row-major `[B, S, D]` buffer into a no-grad tensor.
///
/// Uses `shape.batch/seqlen/dim`; the `hc_mult` axis is dropped. Panics if
/// `data.len() != batch * seqlen * dim`.
pub fn from_flat_bsd(shape: HcShape, data: Vec<f32>) -> Tensor {
    let numel = shape.batch * shape.seqlen * shape.dim;
    assert_eq!(
        data.len(),
        numel,
        "hc_tensor::from_flat_bsd numel mismatch: {} != {}*{}*{}",
        data.len(),
        shape.batch,
        shape.seqlen,
        shape.dim
    );
    Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![shape.batch, shape.seqlen, shape.dim]),
        data,
    ))
}

/// Wrap a flat, row-major `[B, S, HC, D]` buffer into a no-grad tensor.
///
/// Panics if `data.len() != batch * seqlen * hc_mult * dim`.
pub fn from_flat_bshcd(shape: HcShape, data: Vec<f32>) -> Tensor {
    let numel = shape.batch * shape.seqlen * shape.hc_mult * shape.dim;
    assert_eq!(
        data.len(),
        numel,
        "hc_tensor::from_flat_bshcd numel mismatch: {} != {}*{}*{}*{}",
        data.len(),
        shape.batch,
        shape.seqlen,
        shape.hc_mult,
        shape.dim
    );
    Tensor::from_value_no_grad(TensorValue::from_vec(
        Shape(vec![shape.batch, shape.seqlen, shape.hc_mult, shape.dim]),
        data,
    ))
}

/// Expand a `[B, S, D]` tensor across the hyper-connection axis to
/// `[B, S, HC, D]`, replicating each token's contiguous `[D]` slice `hc_mult`
/// times.
///
/// This is the residual-stream entry contract: the embedding is broadcast into
/// every hyper-connection lane. Panics if the input is not 3D.
pub fn expand_hc(x: &Tensor, hc_mult: usize) -> Tensor {
    let value = x.inner.borrow().value.clone();
    let dims = value.shape.0;
    assert_eq!(dims.len(), 3, "hc_tensor::expand_hc input must be [B,S,D]");
    let (b, s, d) = (dims[0], dims[1], dims[2]);
    let mut out = Vec::with_capacity(b * s * hc_mult * d);
    for token in 0..b * s {
        let base = token * d;
        for _ in 0..hc_mult {
            out.extend_from_slice(&value.data[base..base + d]);
        }
    }
    from_flat_bshcd(
        HcShape {
            batch: b,
            seqlen: s,
            hc_mult,
            dim: d,
        },
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bshcd(shape: HcShape, data: Vec<f32>) -> Tensor {
        from_flat_bshcd(shape, data)
    }

    #[test]
    fn to_flat_from_flat_bshcd_round_trips() {
        let shape = HcShape {
            batch: 1,
            seqlen: 2,
            hc_mult: 2,
            dim: 3,
        };
        let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
        let t = bshcd(shape, data.clone());
        assert_eq!(to_flat(&t), data);
        assert_eq!(shape_of(&t), shape);
    }

    #[test]
    fn from_flat_bsd_round_trips_and_drops_hc() {
        let shape = HcShape {
            batch: 2,
            seqlen: 2,
            hc_mult: 9, // ignored by from_flat_bsd
            dim: 3,
        };
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t = from_flat_bsd(shape, data.clone());
        assert_eq!(t.shape().0, vec![2, 2, 3]);
        assert_eq!(to_flat(&t), data);
    }

    #[test]
    fn expand_hc_replicates_token_slices() {
        // [1,2,3]: token0 = [1,2,3], token1 = [4,5,6]
        let x = from_flat_bsd(
            HcShape {
                batch: 1,
                seqlen: 2,
                hc_mult: 1,
                dim: 3,
            },
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        let expanded = expand_hc(&x, 2);
        assert_eq!(expanded.shape().0, vec![1, 2, 2, 3]);
        // each token's [D] slice repeated hc_mult=2 times, contiguously.
        assert_eq!(
            to_flat(&expanded),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    #[should_panic(expected = "numel mismatch")]
    fn from_flat_bshcd_rejects_wrong_numel() {
        from_flat_bshcd(
            HcShape {
                batch: 1,
                seqlen: 2,
                hc_mult: 2,
                dim: 3,
            },
            vec![0.0; 11], // want 12
        );
    }

    #[test]
    #[should_panic(expected = "numel mismatch")]
    fn from_flat_bsd_rejects_wrong_numel() {
        from_flat_bsd(
            HcShape {
                batch: 2,
                seqlen: 2,
                hc_mult: 1,
                dim: 3,
            },
            vec![0.0; 13], // want 12
        );
    }

    #[test]
    #[should_panic(expected = "[B,S,HC,D]")]
    fn shape_of_rejects_non_4d() {
        let t = from_flat_bsd(
            HcShape {
                batch: 1,
                seqlen: 1,
                hc_mult: 1,
                dim: 2,
            },
            vec![0.0, 1.0],
        );
        let _ = shape_of(&t); // t is 3D
    }
}
