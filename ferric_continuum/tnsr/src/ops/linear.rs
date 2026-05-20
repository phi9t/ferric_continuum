//! Dense matrix multiplication with autograd.
//!
//! Book reference: Ch.4 "Matrix Multiplication FLOPs",
//! <https://jax-ml.github.io/scaling-book/transformers/>
//!
//! For Y = X·W where X is (M×K) and W is (K×N):
//!   fwd:   2·M·K·N  FLOPs
//!   ∂L/∂X = (∂L/∂Y)·Wᵀ  →  2·M·K·N  FLOPs  (`raw_linear_dx`)
//!   ∂L/∂W = Xᵀ·(∂L/∂Y)  →  2·M·K·N  FLOPs  (`raw_linear_dw`)
//! Total training cost: 3× fwd per parameter update pass.

use crate::autograd::{BackwardCtx, BackwardRecipe, GradEdge, GradTarget, OpKind};
use crate::saved::{SaveRole, SaveSite, SavedTensor};
use crate::tensor::{Shape, Tensor, TensorValue};

// ---------------------------------------------------------------------------
// Raw matmul helpers: x [B,T,Din] @ w [Din,Dout] -> out [B,T,Dout]
// ---------------------------------------------------------------------------

pub fn raw_linear_forward(x: &TensorValue, w: &TensorValue) -> TensorValue {
    let xd = &x.shape.0;
    let wd = &w.shape.0;

    // x can be 2D [T, Din] or 3D [B, T, Din]
    // w is 2D [Din, Dout]
    assert!(xd.len() >= 2, "linear: x must be at least 2D");
    assert_eq!(wd.len(), 2, "linear: w must be 2D [Din, Dout]");

    let din = *xd.last().unwrap();
    let dout = wd[1];
    assert_eq!(
        wd[0], din,
        "linear: Din mismatch: x has {} but w has {}",
        din, wd[0]
    );

    let batch: usize = xd[..xd.len() - 1].iter().product();
    let mut out_shape = xd[..xd.len() - 1].to_vec();
    out_shape.push(dout);

    let mut out_data = vec![0.0f32; batch * dout];
    let xd_ref = x.data.as_ref();
    let wd_ref = w.data.as_ref();

    for b in 0..batch {
        for j in 0..dout {
            let mut acc = 0.0f32;
            for k in 0..din {
                acc += xd_ref[b * din + k] * wd_ref[k * dout + j];
            }
            out_data[b * dout + j] = acc;
        }
    }

    TensorValue::from_vec(Shape(out_shape), out_data)
}

// dy [B,T,Dout], w [Din,Dout] -> dx [B,T,Din]
pub fn raw_linear_dx(dy: &TensorValue, w: &TensorValue) -> TensorValue {
    let dyd = &dy.shape.0;
    let wd = &w.shape.0;
    let dout = *dyd.last().unwrap();
    let din = wd[0];
    assert_eq!(wd[1], dout);

    let batch: usize = dyd[..dyd.len() - 1].iter().product();
    let mut dx_shape = dyd[..dyd.len() - 1].to_vec();
    dx_shape.push(din);

    let mut dx = vec![0.0f32; batch * din];
    let dyr = dy.data.as_ref();
    let wr = w.data.as_ref();

    for b in 0..batch {
        for k in 0..din {
            let mut acc = 0.0f32;
            for j in 0..dout {
                acc += dyr[b * dout + j] * wr[k * dout + j];
            }
            dx[b * din + k] = acc;
        }
    }

    TensorValue::from_vec(Shape(dx_shape), dx)
}

// x [B,T,Din], dy [B,T,Dout] -> dw [Din,Dout]
pub fn raw_linear_dw(x: &TensorValue, dy: &TensorValue) -> TensorValue {
    let xd = &x.shape.0;
    let dyd = &dy.shape.0;
    let din = *xd.last().unwrap();
    let dout = *dyd.last().unwrap();

    let batch: usize = xd[..xd.len() - 1].iter().product();

    let mut dw = vec![0.0f32; din * dout];
    let xr = x.data.as_ref();
    let dyr = dy.data.as_ref();

    for b in 0..batch {
        for k in 0..din {
            for j in 0..dout {
                dw[k * dout + j] += xr[b * din + k] * dyr[b * dout + j];
            }
        }
    }

    TensorValue::from_vec(Shape(vec![din, dout]), dw)
}

// ---------------------------------------------------------------------------
// linear op
// ---------------------------------------------------------------------------

struct LinearBackward {
    x: SavedTensor,
    w: SavedTensor,
    x_target: GradTarget,
    w_target: GradTarget,
}

impl BackwardRecipe for LinearBackward {
    fn backward(&self, grad_outputs: &[TensorValue], ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        let dy = &grad_outputs[0];
        let x = ctx.unpack(&self.x);
        let w = ctx.unpack(&self.w);

        let dx = raw_linear_dx(dy, &w);
        let dw = raw_linear_dw(&x, dy);

        vec![
            GradEdge {
                target: self.x_target.clone(),
                grad: dx,
            },
            GradEdge {
                target: self.w_target.clone(),
                grad: dw,
            },
        ]
    }
}

pub fn linear(x: &Tensor, w: &Tensor, name: &str) -> Tensor {
    let out_value = raw_linear_forward(&x.inner.borrow().value, &w.inner.borrow().value);

    let need = crate::grad_mode::is_enabled_and_any_requires_grad(&[x, w])
        || crate::checkpoint::is_recording_recompute_saves();

    let (recipe, debug_saved) = if need {
        let x_site = SaveSite::new(
            OpKind::Linear,
            format!("{name}.input"),
            SaveRole::Activation,
            x,
        );
        let w_site = SaveSite::new(
            OpKind::Linear,
            format!("{name}.weight"),
            SaveRole::Parameter,
            w,
        );
        let saved_x = SavedTensor::save(x, x_site.clone());
        let saved_w = SavedTensor::save(w, w_site.clone());

        let recipe: Box<dyn BackwardRecipe> = Box::new(LinearBackward {
            x: saved_x,
            w: saved_w,
            x_target: x.grad_target(),
            w_target: w.grad_target(),
        });
        (Some(recipe), vec![x_site, w_site])
    } else {
        (None, vec![])
    };

    crate::ops::finish_op(
        OpKind::Linear,
        name,
        &[x, w],
        out_value,
        recipe,
        debug_saved,
    )
}
