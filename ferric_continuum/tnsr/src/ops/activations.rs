//! Activation functions: GELU, SiLU, SwiGLU.
//!
//! Book reference: Ch.4 "MLP FLOPs",
//! <https://jax-ml.github.io/scaling-book/transformers/>
//!
//! **`TransformerBlock` uses `gelu`** — the classic 2-matrix GELU MLP
//! (`w1→gelu→w2`).  GELU's FLOP cost is O(B·T·F) via the erf approximation,
//! negligible compared to the surrounding matmuls for large F.
//!
//! **`swiglu`** implements the gated 3-matrix variant the book uses as its
//! reference FFN: `gate = silu(x[..,:F]) * x[...,F:]`, with input shape
//! `[..., 2F]` and output `[..., F]`.  It is NOT wired into `TransformerBlock`
//! but is available for custom blocks using the gated architecture.

use crate::autograd::{BackwardCtx, BackwardRecipe, GradEdge, GradTarget, OpKind};
use crate::saved::{SaveRole, SaveSite, SavedTensor};
use crate::tensor::{Tensor, TensorValue};

// ---------------------------------------------------------------------------
// GELU (exact via erf)
// ---------------------------------------------------------------------------

const SQRT2: f32 = std::f32::consts::SQRT_2;

fn gelu_scalar(x: f32) -> f32 {
    // exact: x * 0.5 * (1 + erf(x / sqrt(2)))
    0.5 * x * (1.0 + erf_approx(x / SQRT2))
}

fn gelu_prime(x: f32) -> f32 {
    // d/dx gelu(x) = 0.5*(1+erf(x/sqrt2)) + x * phi(x)
    // where phi is N(0,1) pdf = exp(-x^2/2)/sqrt(2*pi)
    let phi = (-0.5 * x * x).exp() / (2.0 * std::f32::consts::PI).sqrt();
    0.5 * (1.0 + erf_approx(x / SQRT2)) + x * phi
}

#[allow(clippy::excessive_precision)]
fn erf_approx(x: f32) -> f32 {
    // Abramowitz & Stegun approximation, max error ~1.5e-7
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    sign * (1.0 - poly * (-x * x).exp())
}

fn raw_gelu_forward(x: &TensorValue) -> TensorValue {
    let data: Vec<f32> = x.data.iter().map(|&v| gelu_scalar(v)).collect();
    TensorValue::from_vec(x.shape.clone(), data)
}

fn raw_gelu_backward(dy: &TensorValue, x: &TensorValue) -> TensorValue {
    let data: Vec<f32> = dy
        .data
        .iter()
        .zip(x.data.iter())
        .map(|(&g, &v)| g * gelu_prime(v))
        .collect();
    TensorValue::from_vec(x.shape.clone(), data)
}

struct GeluBackward {
    x: SavedTensor,
    x_target: GradTarget,
}

impl BackwardRecipe for GeluBackward {
    fn backward(&self, grad_outputs: &[TensorValue], ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        let dy = &grad_outputs[0];
        let x = ctx.unpack(&self.x);
        vec![GradEdge {
            target: self.x_target.clone(),
            grad: raw_gelu_backward(dy, &x),
        }]
    }
}

pub fn gelu(x: &Tensor, name: &str) -> Tensor {
    let out_value = raw_gelu_forward(&x.inner.borrow().value);

    let need = crate::grad_mode::is_enabled_and_any_requires_grad(&[x])
        || crate::checkpoint::is_recording_recompute_saves();

    let (recipe, debug_saved) = if need {
        let site = SaveSite::new(
            OpKind::Gelu,
            format!("{name}.input"),
            SaveRole::Activation,
            x,
        );
        let saved = SavedTensor::save(x, site.clone());
        let r: Box<dyn BackwardRecipe> = Box::new(GeluBackward {
            x: saved,
            x_target: x.grad_target(),
        });
        (Some(r), vec![site])
    } else {
        (None, vec![])
    };

    crate::ops::finish_op(OpKind::Gelu, name, &[x], out_value, recipe, debug_saved)
}

// ---------------------------------------------------------------------------
// SiLU
// ---------------------------------------------------------------------------

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

fn silu_scalar(x: f32) -> f32 {
    x * sigmoid(x)
}

fn silu_prime(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

fn raw_silu_forward(x: &TensorValue) -> TensorValue {
    let data: Vec<f32> = x.data.iter().map(|&v| silu_scalar(v)).collect();
    TensorValue::from_vec(x.shape.clone(), data)
}

fn raw_silu_backward(dy: &TensorValue, x: &TensorValue) -> TensorValue {
    let data: Vec<f32> = dy
        .data
        .iter()
        .zip(x.data.iter())
        .map(|(&g, &v)| g * silu_prime(v))
        .collect();
    TensorValue::from_vec(x.shape.clone(), data)
}

struct SiluBackward {
    x: SavedTensor,
    x_target: GradTarget,
}

impl BackwardRecipe for SiluBackward {
    fn backward(&self, grad_outputs: &[TensorValue], ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        let dy = &grad_outputs[0];
        let x = ctx.unpack(&self.x);
        vec![GradEdge {
            target: self.x_target.clone(),
            grad: raw_silu_backward(dy, &x),
        }]
    }
}

pub fn silu(x: &Tensor, name: &str) -> Tensor {
    let out_value = raw_silu_forward(&x.inner.borrow().value);

    let need = crate::grad_mode::is_enabled_and_any_requires_grad(&[x])
        || crate::checkpoint::is_recording_recompute_saves();

    let (recipe, debug_saved) = if need {
        let site = SaveSite::new(
            OpKind::Silu,
            format!("{name}.input"),
            SaveRole::Activation,
            x,
        );
        let saved = SavedTensor::save(x, site.clone());
        let r: Box<dyn BackwardRecipe> = Box::new(SiluBackward {
            x: saved,
            x_target: x.grad_target(),
        });
        (Some(r), vec![site])
    } else {
        (None, vec![])
    };

    crate::ops::finish_op(OpKind::Silu, name, &[x], out_value, recipe, debug_saved)
}

// ---------------------------------------------------------------------------
// SwiGLU: x[..., :D] * silu(x[..., D:])
// Input shape [..., 2*D], output shape [..., D]
// ---------------------------------------------------------------------------

fn raw_swiglu_forward(x: &TensorValue) -> TensorValue {
    let shape = &x.shape.0;
    assert!(
        (*shape.last().unwrap()).is_multiple_of(2),
        "swiglu: last dim must be even"
    );
    let d = shape.last().unwrap() / 2;
    let batch: usize = shape[..shape.len() - 1].iter().product();

    let mut out_data = vec![0.0f32; batch * d];
    let xr = x.data.as_ref();

    for b in 0..batch {
        for i in 0..d {
            let gate_val = xr[b * (2 * d) + i];
            let up_val = xr[b * (2 * d) + d + i];
            out_data[b * d + i] = gate_val * silu_scalar(up_val);
        }
    }

    let mut out_shape = shape[..shape.len() - 1].to_vec();
    out_shape.push(d);
    TensorValue::from_vec(crate::tensor::Shape(out_shape), out_data)
}

fn raw_swiglu_backward(dy: &TensorValue, x: &TensorValue) -> TensorValue {
    let shape = &x.shape.0;
    let d = shape.last().unwrap() / 2;
    let batch: usize = shape[..shape.len() - 1].iter().product();

    let mut dx = vec![0.0f32; x.shape.numel()];
    let xr = x.data.as_ref();
    let dyr = dy.data.as_ref();

    for b in 0..batch {
        for i in 0..d {
            let gate_val = xr[b * (2 * d) + i];
            let up_val = xr[b * (2 * d) + d + i];
            let dy_i = dyr[b * d + i];

            // d(gate * silu(up))/d_gate = silu(up)
            // d(gate * silu(up))/d_up   = gate * silu'(up)
            dx[b * (2 * d) + i] = dy_i * silu_scalar(up_val);
            dx[b * (2 * d) + d + i] = dy_i * gate_val * silu_prime(up_val);
        }
    }

    TensorValue::from_vec(x.shape.clone(), dx)
}

struct SwiGluBackward {
    x: SavedTensor,
    x_target: GradTarget,
}

impl BackwardRecipe for SwiGluBackward {
    fn backward(&self, grad_outputs: &[TensorValue], ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        let dy = &grad_outputs[0];
        let x = ctx.unpack(&self.x);
        vec![GradEdge {
            target: self.x_target.clone(),
            grad: raw_swiglu_backward(dy, &x),
        }]
    }
}

pub fn swiglu(x: &Tensor, name: &str) -> Tensor {
    let out_value = raw_swiglu_forward(&x.inner.borrow().value);

    let need = crate::grad_mode::is_enabled_and_any_requires_grad(&[x])
        || crate::checkpoint::is_recording_recompute_saves();

    let (recipe, debug_saved) = if need {
        let site = SaveSite::new(
            OpKind::SwiGlu,
            format!("{name}.input"),
            SaveRole::Activation,
            x,
        );
        let saved = SavedTensor::save(x, site.clone());
        let r: Box<dyn BackwardRecipe> = Box::new(SwiGluBackward {
            x: saved,
            x_target: x.grad_target(),
        });
        (Some(r), vec![site])
    } else {
        (None, vec![])
    };

    crate::ops::finish_op(OpKind::SwiGlu, name, &[x], out_value, recipe, debug_saved)
}
