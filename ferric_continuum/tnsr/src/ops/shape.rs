//! Shape-only ops: `reshape`.
//!
//! Reshape preserves the underlying flat buffer (row-major) and only changes
//! the logical shape.  It is the autograd-aware bridge between the flat
//! `[B,T,D]` activation layout used by `linear` and the multi-head
//! `[B,T,H,Dh]` layout required by per-head RMSNorm, RoPE, and GQA.

use crate::autograd::{BackwardCtx, BackwardRecipe, GradEdge, GradTarget, OpKind};
use crate::tensor::{Shape, Tensor, TensorValue};

fn raw_reshape(x: &TensorValue, new_shape: Shape) -> TensorValue {
    assert_eq!(
        x.shape.numel(),
        new_shape.numel(),
        "reshape: numel mismatch {:?} -> {:?}",
        x.shape.0,
        new_shape.0
    );
    TensorValue {
        shape: new_shape,
        data: x.data.clone(),
    }
}

struct ReshapeBackward {
    x_target: GradTarget,
    orig_shape: Shape,
}

impl BackwardRecipe for ReshapeBackward {
    fn backward(&self, grad_outputs: &[TensorValue], _ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        let dy = &grad_outputs[0];
        let grad = TensorValue {
            shape: self.orig_shape.clone(),
            data: dy.data.clone(),
        };
        vec![GradEdge {
            target: self.x_target.clone(),
            grad,
        }]
    }
}

pub fn reshape(x: &Tensor, new_shape: &[usize], name: &str) -> Tensor {
    let shape = Shape(new_shape.to_vec());
    let orig_shape = x.inner.borrow().value.shape.clone();
    let out_value = raw_reshape(&x.inner.borrow().value, shape);

    let recipe: Option<Box<dyn BackwardRecipe>> =
        if crate::grad_mode::is_enabled_and_any_requires_grad(&[x]) {
            Some(Box::new(ReshapeBackward {
                x_target: x.grad_target(),
                orig_shape,
            }))
        } else {
            None
        };

    crate::ops::finish_op(OpKind::Reshape, name, &[x], out_value, recipe, vec![])
}

// ---------------------------------------------------------------------------
// split3 — the canonical MULTI-OUTPUT op
// ---------------------------------------------------------------------------
//
// One tensor in, three tensors out: splits the last axis into 3 equal chunks.
// This is the fused-QKV pattern: `linear(x, w_qkv)` gives `[..., 3*D]`, and
// `split3` peels it into `q, k, v`, each `[..., D]`.
//
// Forward: copy disjoint slices along the last axis.
// Backward: the engine hands us THREE incoming gradients (∂L/∂q, ∂L/∂k, ∂L/∂v)
//           in `grad_outputs[0..3]`; since each output is a plain slice, the
//           input gradient is just their concatenation back to `[..., 3*D]`.
//           (`∂L/∂x = Σᵢ (∂yᵢ/∂x)ᵀ·∂L/∂yᵢ`, and here each `∂yᵢ/∂x` is a slice
//           selector, so the sum is a concat into disjoint regions.)

/// Split the last axis of `x` into 3 equal parts along the innermost dimension.
/// The concatenation of the returned parts equals `x`.
fn raw_split3(x: &TensorValue) -> [TensorValue; 3] {
    let dims = &x.shape.0;
    let last = *dims.last().expect("split3: input must have >= 1 dim");
    assert_eq!(last % 3, 0, "split3: last dim {last} not divisible by 3");
    let chunk = last / 3;
    let rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);

    let mut out_shape = dims.clone();
    *out_shape.last_mut().unwrap() = chunk;
    let out_shape = Shape(out_shape);

    let src = x.data.as_ref();
    let mut parts: Vec<Vec<f32>> =
        (0..3).map(|_| Vec::with_capacity(rows * chunk)).collect();
    for r in 0..rows {
        let base = r * last;
        for (p, part) in parts.iter_mut().enumerate() {
            let start = base + p * chunk;
            part.extend_from_slice(&src[start..start + chunk]);
        }
    }

    let mut it = parts.into_iter();
    let q = TensorValue::from_vec(out_shape.clone(), it.next().unwrap());
    let k = TensorValue::from_vec(out_shape.clone(), it.next().unwrap());
    let v = TensorValue::from_vec(out_shape, it.next().unwrap());
    [q, k, v]
}

/// Concatenate three equal-shaped tensors along the last axis: the exact
/// inverse of `raw_split3`, used to reassemble the input gradient in backward.
fn raw_concat3_last(parts: &[&TensorValue; 3]) -> TensorValue {
    let dims = &parts[0].shape.0;
    let chunk = *dims.last().expect("concat3: parts must have >= 1 dim");
    let rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);

    let mut out_shape = dims.clone();
    *out_shape.last_mut().unwrap() = chunk * 3;
    let out_shape = Shape(out_shape);

    let mut data = Vec::with_capacity(rows * chunk * 3);
    for r in 0..rows {
        for part in parts {
            let base = r * chunk;
            data.extend_from_slice(&part.data.as_ref()[base..base + chunk]);
        }
    }
    TensorValue::from_vec(out_shape, data)
}

struct Split3Backward {
    x_target: GradTarget,
}

impl BackwardRecipe for Split3Backward {
    fn backward(&self, grad_outputs: &[TensorValue], _ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        // grad_outputs = [∂L/∂q, ∂L/∂k, ∂L/∂v]; concat back to [..., 3*D].
        let grad = raw_concat3_last(&[&grad_outputs[0], &grad_outputs[1], &grad_outputs[2]]);
        vec![GradEdge {
            target: self.x_target.clone(),
            grad,
        }]
    }
}

/// Split `x` (`[..., 3*D]`) into three `[..., D]` tensors along the last axis.
/// Concatenating the outputs reproduces `x`. This is a multi-output op: all
/// three outputs share one `OpCall`, and backward concatenates their gradients.
pub fn split3(x: &Tensor, name: &str) -> (Tensor, Tensor, Tensor) {
    let [q_val, k_val, v_val] = raw_split3(&x.inner.borrow().value);

    let recipe: Option<Box<dyn BackwardRecipe>> =
        if crate::grad_mode::is_enabled_and_any_requires_grad(&[x]) {
            Some(Box::new(Split3Backward {
                x_target: x.grad_target(),
            }))
        } else {
            None
        };

    let mut outs = crate::ops::finish_op_multi(
        OpKind::Split,
        name,
        &[x],
        vec![q_val, k_val, v_val],
        recipe,
        vec![],
    );
    let v = outs.pop().unwrap();
    let k = outs.pop().unwrap();
    let q = outs.pop().unwrap();
    (q, k, v)
}
