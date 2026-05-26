//! Rotary Position Embedding (RoPE).
//!
//! Used by Qwen2 / Qwen3 (and Llama).  Applied to Q and K per-head, after the
//! Q/K projection (and, in Qwen3, after the per-head Q/K RMSNorm).
//!
//! Input/output shape: `[B, T, H, Dh]` with `Dh` even.  Each 2-vector pair
//! `(x[2i], x[2i+1])` is rotated by `θ = pos · base^(-2i/Dh)`:
//!
//! ```text
//! y[2i]   = x[2i] · cos(θ) - x[2i+1] · sin(θ)
//! y[2i+1] = x[2i] · sin(θ) + x[2i+1] · cos(θ)
//! ```
//!
//! Backward applies the transpose (= inverse) rotation `R(-θ)`:
//!
//! ```text
//! dx[2i]   = dy[2i]   · cos(θ) + dy[2i+1] · sin(θ)
//! dx[2i+1] = -dy[2i]  · sin(θ) + dy[2i+1] · cos(θ)
//! ```
//!
//! Qwen3 uses `rope_theta = 1_000_000` per the technical report; pick the base
//! freely via `RopeConfig`.

use crate::autograd::{BackwardCtx, BackwardRecipe, GradEdge, GradTarget, OpKind};
use crate::tensor::{Tensor, TensorValue};

#[derive(Clone, Copy, Debug)]
pub struct RopeConfig {
    pub base: f32,
    pub start_pos: usize,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            base: 1_000_000.0,
            start_pos: 0,
        }
    }
}

fn cos_sin_table(seq: usize, head_dim: usize, cfg: RopeConfig) -> (Vec<f32>, Vec<f32>) {
    assert!(head_dim % 2 == 0, "rope: head_dim must be even");
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; seq * half];
    let mut sin = vec![0.0f32; seq * half];
    let inv = |i: usize| -> f32 { cfg.base.powf(-(2.0 * i as f32) / head_dim as f32) };
    for t in 0..seq {
        let pos = (cfg.start_pos + t) as f32;
        for i in 0..half {
            let theta = pos * inv(i);
            cos[t * half + i] = theta.cos();
            sin[t * half + i] = theta.sin();
        }
    }
    (cos, sin)
}

fn raw_rope_forward(x: &TensorValue, cos: &[f32], sin: &[f32]) -> TensorValue {
    let sh = &x.shape.0;
    assert_eq!(sh.len(), 4, "rope: expected 4D [B,T,H,Dh]");
    let (b, t, h, dh) = (sh[0], sh[1], sh[2], sh[3]);
    let half = dh / 2;
    let xr = x.data.as_ref();
    let mut out = vec![0.0f32; b * t * h * dh];

    for bi in 0..b {
        for ti in 0..t {
            for hi in 0..h {
                let base_in = ((bi * t + ti) * h + hi) * dh;
                let base_cs = ti * half;
                for i in 0..half {
                    let c = cos[base_cs + i];
                    let s = sin[base_cs + i];
                    let x0 = xr[base_in + 2 * i];
                    let x1 = xr[base_in + 2 * i + 1];
                    out[base_in + 2 * i] = x0 * c - x1 * s;
                    out[base_in + 2 * i + 1] = x0 * s + x1 * c;
                }
            }
        }
    }

    TensorValue::from_vec(x.shape.clone(), out)
}

fn raw_rope_backward(dy: &TensorValue, cos: &[f32], sin: &[f32]) -> TensorValue {
    let sh = &dy.shape.0;
    let (b, t, h, dh) = (sh[0], sh[1], sh[2], sh[3]);
    let half = dh / 2;
    let dyr = dy.data.as_ref();
    let mut dx = vec![0.0f32; b * t * h * dh];

    for bi in 0..b {
        for ti in 0..t {
            for hi in 0..h {
                let base = ((bi * t + ti) * h + hi) * dh;
                let base_cs = ti * half;
                for i in 0..half {
                    let c = cos[base_cs + i];
                    let s = sin[base_cs + i];
                    let g0 = dyr[base + 2 * i];
                    let g1 = dyr[base + 2 * i + 1];
                    dx[base + 2 * i] = g0 * c + g1 * s;
                    dx[base + 2 * i + 1] = -g0 * s + g1 * c;
                }
            }
        }
    }

    TensorValue::from_vec(dy.shape.clone(), dx)
}

struct RopeBackward {
    cos: Vec<f32>,
    sin: Vec<f32>,
    x_target: GradTarget,
}

impl BackwardRecipe for RopeBackward {
    fn backward(&self, grad_outputs: &[TensorValue], _ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        let dy = &grad_outputs[0];
        vec![GradEdge {
            target: self.x_target.clone(),
            grad: raw_rope_backward(dy, &self.cos, &self.sin),
        }]
    }
}

/// Apply RoPE to `x` of shape `[B, T, H, Dh]`.
pub fn rope(x: &Tensor, cfg: RopeConfig, name: &str) -> Tensor {
    let (seq, head_dim) = {
        let xv = x.inner.borrow();
        let sh = &xv.value.shape.0;
        assert_eq!(sh.len(), 4, "rope: expected 4D [B,T,H,Dh]");
        (sh[1], sh[3])
    };
    let (cos, sin) = cos_sin_table(seq, head_dim, cfg);
    let out_value = raw_rope_forward(&x.inner.borrow().value, &cos, &sin);

    let recipe: Option<Box<dyn BackwardRecipe>> =
        if crate::grad_mode::is_enabled_and_any_requires_grad(&[x]) {
            Some(Box::new(RopeBackward {
                cos,
                sin,
                x_target: x.grad_target(),
            }))
        } else {
            None
        };

    crate::ops::finish_op(OpKind::Rope, name, &[x], out_value, recipe, vec![])
}
