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
