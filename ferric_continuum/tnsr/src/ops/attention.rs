//! Attention scores, softmax, mix, and causal mask.
//!
//! Book reference: Ch.4 "Attention FLOPs",
//! <https://jax-ml.github.io/scaling-book/transformers/>
//!
//! Using tnsr notation (N=1 head, H=D, so shapes are `[B,T,D]`):
//!   attn_scores: Q·Kᵀ  →  2·B·T²·D  FLOPs  (`attention_scores`)
//!   attn_mix:    P·V   →  2·B·T²·D  FLOPs  (`attention_mix`)
//! Combined attention cost: 4·B·T²·D fwd, 8·B·T²·D bwd.
//!
//! **Causal mask note:** tnsr materializes the full `[B,T,T]` tensor and
//! overwrites the upper triangle — O(B·T²) writes, no FMA arithmetic.
//! A fused kernel could skip the upper triangle, but tnsr pays the full cost.
//!
//! In the book's multi-head notation: replace D with N·H and the formulas
//! become 2·B·T²·N·H.  In tnsr, N=1 and H=D, so N·H = D.

use crate::autograd::{BackwardCtx, BackwardRecipe, GradEdge, GradTarget, OpKind};
use crate::saved::{SaveRole, SaveSite, SavedTensor};
use crate::tensor::{Shape, Tensor, TensorValue};

// ---------------------------------------------------------------------------
// attention_scores: Q [B,T,D] x K [B,T,D] -> scores [B,T,T]
// scores[b,t,s] = sum_d Q[b,t,d] * K[b,s,d] * scale
// ---------------------------------------------------------------------------

fn raw_attention_scores(q: &TensorValue, k: &TensorValue, scale: f32) -> TensorValue {
    let qd = &q.shape.0;
    assert_eq!(qd.len(), 3, "Q must be 3D [B,T,D]");
    let (b, t, d) = (qd[0], qd[1], qd[2]);

    let kd = &k.shape.0;
    assert_eq!((kd[0], kd[1], kd[2]), (b, t, d), "K shape must match Q");

    let qr = q.data.as_ref();
    let kr = k.data.as_ref();
    let mut out = vec![0.0f32; b * t * t];

    for bi in 0..b {
        for ti in 0..t {
            for si in 0..t {
                let mut acc = 0.0f32;
                for di in 0..d {
                    acc += qr[bi * t * d + ti * d + di] * kr[bi * t * d + si * d + di];
                }
                out[bi * t * t + ti * t + si] = acc * scale;
            }
        }
    }

    TensorValue::from_vec(Shape(vec![b, t, t]), out)
}

// dQ [B,T,D], dK [B,T,D]
fn raw_attention_scores_backward(
    dscores: &TensorValue,
    q: &TensorValue,
    k: &TensorValue,
    scale: f32,
) -> (TensorValue, TensorValue) {
    let qd = &q.shape.0;
    let (b, t, d) = (qd[0], qd[1], qd[2]);

    let dsr = dscores.data.as_ref();
    let qr = q.data.as_ref();
    let kr = k.data.as_ref();

    let mut dq = vec![0.0f32; b * t * d];
    let mut dk = vec![0.0f32; b * t * d];

    for bi in 0..b {
        for ti in 0..t {
            for si in 0..t {
                let ds = dsr[bi * t * t + ti * t + si] * scale;
                for di in 0..d {
                    dq[bi * t * d + ti * d + di] += ds * kr[bi * t * d + si * d + di];
                    dk[bi * t * d + si * d + di] += ds * qr[bi * t * d + ti * d + di];
                }
            }
        }
    }

    (
        TensorValue::from_vec(q.shape.clone(), dq),
        TensorValue::from_vec(k.shape.clone(), dk),
    )
}

struct AttentionScoresBackward {
    q: SavedTensor,
    k: SavedTensor,
    q_target: GradTarget,
    k_target: GradTarget,
    scale: f32,
}

impl BackwardRecipe for AttentionScoresBackward {
    fn backward(&self, grad_outputs: &[TensorValue], ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        let dscores = &grad_outputs[0];
        let q = ctx.unpack(&self.q);
        let k = ctx.unpack(&self.k);
        let (dq, dk) = raw_attention_scores_backward(dscores, &q, &k, self.scale);
        vec![
            GradEdge {
                target: self.q_target.clone(),
                grad: dq,
            },
            GradEdge {
                target: self.k_target.clone(),
                grad: dk,
            },
        ]
    }
}

pub fn attention_scores(q: &Tensor, k: &Tensor, scale: f32, name: &str) -> Tensor {
    let out_value = raw_attention_scores(&q.inner.borrow().value, &k.inner.borrow().value, scale);

    let need = crate::grad_mode::is_enabled_and_any_requires_grad(&[q, k])
        || crate::checkpoint::is_recording_recompute_saves();

    let (recipe, debug_saved) = if need {
        let q_site = SaveSite::new(
            OpKind::AttentionScores,
            format!("{name}.q"),
            SaveRole::Activation,
            q,
        );
        let k_site = SaveSite::new(
            OpKind::AttentionScores,
            format!("{name}.k"),
            SaveRole::Activation,
            k,
        );
        let saved_q = SavedTensor::save(q, q_site.clone());
        let saved_k = SavedTensor::save(k, k_site.clone());
        let r: Box<dyn BackwardRecipe> = Box::new(AttentionScoresBackward {
            q: saved_q,
            k: saved_k,
            q_target: q.grad_target(),
            k_target: k.grad_target(),
            scale,
        });
        (Some(r), vec![q_site, k_site])
    } else {
        (None, vec![])
    };

    crate::ops::finish_op(
        OpKind::AttentionScores,
        name,
        &[q, k],
        out_value,
        recipe,
        debug_saved,
    )
}

// ---------------------------------------------------------------------------
// softmax_last_dim: x [B,T,T] -> p [B,T,T]
// ---------------------------------------------------------------------------

/// CPU row-softmax used by the default path, CUDA fallback, and agreement tests.
pub fn raw_softmax_cpu(x: &TensorValue) -> TensorValue {
    let xd = &x.shape.0;
    let d = *xd.last().unwrap();
    let batch: usize = xd[..xd.len() - 1].iter().product();
    let xr = x.data.as_ref();

    let mut out = vec![0.0f32; x.shape.numel()];

    for b in 0..batch {
        let row = &xr[b * d..(b + 1) * d];
        let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - max_v).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for i in 0..d {
            out[b * d + i] = exps[i] / sum;
        }
    }

    TensorValue::from_vec(x.shape.clone(), out)
}

pub fn raw_softmax(x: &TensorValue) -> TensorValue {
    let xd = &x.shape.0;
    let d = *xd.last().unwrap();
    let batch: usize = xd[..xd.len() - 1].iter().product();
    let xr = x.data.as_ref();

    // Optional GPU forward (host in / host out); backward remains CPU on host data.
    if crate::cuda_ffi::use_cuda() {
        match crate::cuda_ffi::softmax_f32(batch, d, xr) {
            Some(out) => return TensorValue::from_vec(x.shape.clone(), out),
            None => {
                eprintln!(
                    "tnsr: CUDA softmax failed (rows={batch}, cols={d}); \
                     falling back to CPU"
                );
            }
        }
    }

    raw_softmax_cpu(x)
}

fn raw_softmax_backward(dy: &TensorValue, p: &TensorValue) -> TensorValue {
    let pd = &p.shape.0;
    let d = *pd.last().unwrap();
    let batch: usize = pd[..pd.len() - 1].iter().product();
    let dyr = dy.data.as_ref();
    let pr = p.data.as_ref();

    let mut dx = vec![0.0f32; p.shape.numel()];

    for b in 0..batch {
        // dot = sum_j dy[j] * p[j]
        let dot: f32 = (0..d).map(|j| dyr[b * d + j] * pr[b * d + j]).sum();
        for i in 0..d {
            dx[b * d + i] = pr[b * d + i] * (dyr[b * d + i] - dot);
        }
    }

    TensorValue::from_vec(p.shape.clone(), dx)
}

struct SoftmaxBackward {
    p: SavedTensor,
    x_target: GradTarget,
}

impl BackwardRecipe for SoftmaxBackward {
    fn backward(&self, grad_outputs: &[TensorValue], ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        let dy = &grad_outputs[0];
        let p = ctx.unpack(&self.p);
        vec![GradEdge {
            target: self.x_target.clone(),
            grad: raw_softmax_backward(dy, &p),
        }]
    }
}

pub fn softmax_last_dim(x: &Tensor, name: &str) -> Tensor {
    let out_value = raw_softmax(&x.inner.borrow().value);

    let need = crate::grad_mode::is_enabled_and_any_requires_grad(&[x])
        || crate::checkpoint::is_recording_recompute_saves();

    let (recipe, debug_saved) = if need {
        // Save output (p), not input, since backward only needs p
        let site = SaveSite::new(
            OpKind::Softmax,
            format!("{name}.output"),
            SaveRole::Activation,
            x,
        );
        // We need to save the output, but we're computing from x. Create a temp tensor for the output.
        let out_tensor = Tensor::from_value_no_grad(out_value.clone());
        let saved_p = SavedTensor::save(&out_tensor, site.clone());
        let r: Box<dyn BackwardRecipe> = Box::new(SoftmaxBackward {
            p: saved_p,
            x_target: x.grad_target(),
        });
        (Some(r), vec![site])
    } else {
        (None, vec![])
    };

    crate::ops::finish_op(OpKind::Softmax, name, &[x], out_value, recipe, debug_saved)
}

// ---------------------------------------------------------------------------
// attention_mix: P [B,T,T] x V [B,T,D] -> out [B,T,D]
// out[b,t,d] = sum_s P[b,t,s] * V[b,s,d]
// ---------------------------------------------------------------------------

fn raw_attention_mix(p: &TensorValue, v: &TensorValue) -> TensorValue {
    let vd = &v.shape.0;
    let (b, t, d) = (vd[0], vd[1], vd[2]);
    assert_eq!(p.shape.0, vec![b, t, t], "P shape must be [B,T,T]");

    let pr = p.data.as_ref();
    let vr = v.data.as_ref();
    let mut out = vec![0.0f32; b * t * d];

    for bi in 0..b {
        for ti in 0..t {
            for di in 0..d {
                let mut acc = 0.0f32;
                for si in 0..t {
                    acc += pr[bi * t * t + ti * t + si] * vr[bi * t * d + si * d + di];
                }
                out[bi * t * d + ti * d + di] = acc;
            }
        }
    }

    TensorValue::from_vec(Shape(vec![b, t, d]), out)
}

fn raw_attention_mix_backward(
    dout: &TensorValue,
    p: &TensorValue,
    v: &TensorValue,
) -> (TensorValue, TensorValue) {
    let vd = &v.shape.0;
    let (b, t, d) = (vd[0], vd[1], vd[2]);

    let doutr = dout.data.as_ref();
    let pr = p.data.as_ref();
    let vr = v.data.as_ref();

    let mut dp = vec![0.0f32; b * t * t];
    let mut dv = vec![0.0f32; b * t * d];

    for bi in 0..b {
        for ti in 0..t {
            for si in 0..t {
                let mut dp_acc = 0.0f32;
                for di in 0..d {
                    let do_val = doutr[bi * t * d + ti * d + di];
                    dp_acc += do_val * vr[bi * t * d + si * d + di];
                    dv[bi * t * d + si * d + di] += do_val * pr[bi * t * t + ti * t + si];
                }
                dp[bi * t * t + ti * t + si] = dp_acc;
            }
        }
    }

    (
        TensorValue::from_vec(p.shape.clone(), dp),
        TensorValue::from_vec(v.shape.clone(), dv),
    )
}

struct AttentionMixBackward {
    p: SavedTensor,
    v: SavedTensor,
    p_target: GradTarget,
    v_target: GradTarget,
}

impl BackwardRecipe for AttentionMixBackward {
    fn backward(&self, grad_outputs: &[TensorValue], ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        let dout = &grad_outputs[0];
        let p = ctx.unpack(&self.p);
        let v = ctx.unpack(&self.v);
        let (dp, dv) = raw_attention_mix_backward(dout, &p, &v);
        vec![
            GradEdge {
                target: self.p_target.clone(),
                grad: dp,
            },
            GradEdge {
                target: self.v_target.clone(),
                grad: dv,
            },
        ]
    }
}

pub fn attention_mix(p: &Tensor, v: &Tensor, name: &str) -> Tensor {
    let out_value = raw_attention_mix(&p.inner.borrow().value, &v.inner.borrow().value);

    let need = crate::grad_mode::is_enabled_and_any_requires_grad(&[p, v])
        || crate::checkpoint::is_recording_recompute_saves();

    let (recipe, debug_saved) = if need {
        let p_site = SaveSite::new(
            OpKind::AttentionMix,
            format!("{name}.p"),
            SaveRole::Activation,
            p,
        );
        let v_site = SaveSite::new(
            OpKind::AttentionMix,
            format!("{name}.v"),
            SaveRole::Activation,
            v,
        );
        let saved_p = SavedTensor::save(p, p_site.clone());
        let saved_v = SavedTensor::save(v, v_site.clone());
        let r: Box<dyn BackwardRecipe> = Box::new(AttentionMixBackward {
            p: saved_p,
            v: saved_v,
            p_target: p.grad_target(),
            v_target: v.grad_target(),
        });
        (Some(r), vec![p_site, v_site])
    } else {
        (None, vec![])
    };

    crate::ops::finish_op(
        OpKind::AttentionMix,
        name,
        &[p, v],
        out_value,
        recipe,
        debug_saved,
    )
}

// ---------------------------------------------------------------------------
// causal mask: set upper triangle (s > t) to -inf before softmax
// ---------------------------------------------------------------------------

pub fn causal_mask(scores: &Tensor, name: &str) -> Tensor {
    let sv = scores.inner.borrow().value.clone();
    let sd = &sv.shape.0;
    assert!(sd.len() >= 2);
    let t = *sd.last().unwrap();
    let mut data = sv.data.as_ref().clone();
    // Treat shape as [..., T, T]; iterate over all query positions (b_t = product of leading dims)
    let b_t = sd[..sd.len() - 1].iter().product::<usize>();

    for row_i in 0..b_t {
        let ti = row_i % t; // position in sequence
        for si in 0..t {
            if si > ti {
                data[row_i * t + si] = f32::NEG_INFINITY;
            }
        }
    }

    let out_value = TensorValue::from_vec(sv.shape.clone(), data);

    // Causal mask doesn't contribute gradients (it's a constant mask)
    // Pass through gradient as identity (masked positions get -inf in forward,
    // exp(-inf)=0 so their gradient is naturally zero through softmax)
    let recipe: Option<Box<dyn BackwardRecipe>> =
        if crate::grad_mode::is_enabled_and_any_requires_grad(&[scores]) {
            Some(Box::new(IdentityBackward {
                target: scores.grad_target(),
            }))
        } else {
            None
        };

    crate::ops::finish_op(
        OpKind::CausalMask,
        name,
        &[scores],
        out_value,
        recipe,
        vec![],
    )
}

struct IdentityBackward {
    target: GradTarget,
}

impl BackwardRecipe for IdentityBackward {
    fn backward(&self, grad_outputs: &[TensorValue], _ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        vec![GradEdge {
            target: self.target.clone(),
            grad: grad_outputs[0].clone(),
        }]
    }
}
