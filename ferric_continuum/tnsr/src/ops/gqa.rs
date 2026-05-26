//! Grouped Query Attention (GQA) with causal masking.
//!
//! Single fused op: takes pre-projected (and, for Q3, RoPE'd + QK-normed)
//!   Q : `[B, T, Hq, Dh]`
//!   K : `[B, T, Hk, Dh]`
//!   V : `[B, T, Hk, Dh]`
//! and returns
//!   O : `[B, T, Hq, Dh]`
//!
//! Each query head `qh` attends to KV-head `kh = qh / (Hq/Hk)`.  Scores are
//! scaled by `1/sqrt(Dh)` and causally masked (positions `s > t` are forbidden).
//!
//! The op fuses scores → mask → softmax → mix so that only the softmax output
//! `P [B, Hq, T, T]` needs to be saved for backward (no separate scores tensor).
//!
//! FLOPs (per layer): `4 · B · T² · Hq · Dh` forward, `8 · B · T² · Hq · Dh`
//! backward — the same `4 B T² N H` formula as classic MHA, with `N = Hq`.

use crate::autograd::{BackwardCtx, BackwardRecipe, GradEdge, GradTarget, OpKind};
use crate::tensor::{Shape, Tensor, TensorValue};

#[derive(Clone, Copy, Debug)]
pub struct GqaShape {
    pub b: usize,
    pub t: usize,
    pub hq: usize,
    pub hk: usize,
    pub dh: usize,
}

impl GqaShape {
    pub fn group_size(&self) -> usize {
        assert!(self.hq % self.hk == 0, "Hq must be divisible by Hk");
        self.hq / self.hk
    }
}

fn check_shape_qkv(q: &TensorValue, k: &TensorValue, v: &TensorValue) -> GqaShape {
    let qs = &q.shape.0;
    let ks = &k.shape.0;
    let vs = &v.shape.0;
    assert_eq!(qs.len(), 4, "gqa: Q must be [B,T,Hq,Dh]");
    assert_eq!(ks.len(), 4, "gqa: K must be [B,T,Hk,Dh]");
    assert_eq!(vs.len(), 4, "gqa: V must be [B,T,Hk,Dh]");
    assert_eq!(qs[0], ks[0]);
    assert_eq!(qs[0], vs[0]);
    assert_eq!(qs[1], ks[1]);
    assert_eq!(qs[1], vs[1]);
    assert_eq!(ks[2], vs[2]);
    assert_eq!(qs[3], ks[3]);
    assert_eq!(qs[3], vs[3]);
    GqaShape {
        b: qs[0],
        t: qs[1],
        hq: qs[2],
        hk: ks[2],
        dh: qs[3],
    }
}

fn raw_gqa_forward(
    q: &TensorValue,
    k: &TensorValue,
    v: &TensorValue,
    sh: GqaShape,
) -> (TensorValue, Vec<f32>) {
    let (b, t, hq, hk, dh) = (sh.b, sh.t, sh.hq, sh.hk, sh.dh);
    let gs = sh.group_size();
    let scale = (dh as f32).sqrt().recip();

    let qr = q.data.as_ref();
    let kr = k.data.as_ref();
    let vr = v.data.as_ref();
    let mut out = vec![0.0f32; b * t * hq * dh];
    let mut p_all = vec![0.0f32; b * hq * t * t];

    let q_off = |bi, ti, hi| ((bi * t + ti) * hq + hi) * dh;
    let kv_off = |bi, ti, hi| ((bi * t + ti) * hk + hi) * dh;
    let p_off = |bi, hi, ti| ((bi * hq + hi) * t + ti) * t;

    for bi in 0..b {
        for hi in 0..hq {
            let kh = hi / gs;
            for ti in 0..t {
                // scores[s] = (sum_d Q[b,t,hi,d] * K[b,s,kh,d]) * scale
                let mut row = vec![f32::NEG_INFINITY; t];
                let qi = q_off(bi, ti, hi);
                let mut max_v = f32::NEG_INFINITY;
                for si in 0..=ti {
                    let ki = kv_off(bi, si, kh);
                    let mut acc = 0.0f32;
                    for d in 0..dh {
                        acc += qr[qi + d] * kr[ki + d];
                    }
                    let s = acc * scale;
                    row[si] = s;
                    if s > max_v {
                        max_v = s;
                    }
                }
                let mut sum = 0.0f32;
                for si in 0..=ti {
                    let e = (row[si] - max_v).exp();
                    row[si] = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                let po = p_off(bi, hi, ti);
                for si in 0..t {
                    let p = if si <= ti { row[si] * inv } else { 0.0 };
                    p_all[po + si] = p;
                }
                // mix: out[..,d] = sum_s p[s] * V[b,s,kh,d]
                let oi = q_off(bi, ti, hi);
                for d in 0..dh {
                    let mut acc = 0.0f32;
                    for si in 0..=ti {
                        let vi = kv_off(bi, si, kh);
                        acc += p_all[po + si] * vr[vi + d];
                    }
                    out[oi + d] = acc;
                }
            }
        }
    }

    (TensorValue::from_vec(Shape(vec![b, t, hq, dh]), out), p_all)
}

fn raw_gqa_backward(
    dout: &TensorValue,
    q: &TensorValue,
    k: &TensorValue,
    v: &TensorValue,
    p_all: &[f32],
    sh: GqaShape,
) -> (TensorValue, TensorValue, TensorValue) {
    let (b, t, hq, hk, dh) = (sh.b, sh.t, sh.hq, sh.hk, sh.dh);
    let gs = sh.group_size();
    let scale = (dh as f32).sqrt().recip();

    let qr = q.data.as_ref();
    let kr = k.data.as_ref();
    let vr = v.data.as_ref();
    let doutr = dout.data.as_ref();

    let mut dq = vec![0.0f32; b * t * hq * dh];
    let mut dk = vec![0.0f32; b * t * hk * dh];
    let mut dv = vec![0.0f32; b * t * hk * dh];

    let q_off = |bi, ti, hi| ((bi * t + ti) * hq + hi) * dh;
    let kv_off = |bi, ti, hi| ((bi * t + ti) * hk + hi) * dh;
    let p_off = |bi, hi, ti| ((bi * hq + hi) * t + ti) * t;

    for bi in 0..b {
        for hi in 0..hq {
            let kh = hi / gs;
            for ti in 0..t {
                let po = p_off(bi, hi, ti);
                let oi = q_off(bi, ti, hi);

                // dP[s] = sum_d dout[..,d] * V[b,s,kh,d]
                // dV[b,s,kh,d] += P[s] * dout[..,d]
                let mut dp = vec![0.0f32; t];
                for si in 0..=ti {
                    let vi = kv_off(bi, si, kh);
                    let mut acc = 0.0f32;
                    for d in 0..dh {
                        let g = doutr[oi + d];
                        acc += g * vr[vi + d];
                        dv[vi + d] += p_all[po + si] * g;
                    }
                    dp[si] = acc;
                }

                // softmax backward: dS[i] = P[i] * (dP[i] - sum_j P[j] * dP[j])
                let mut dot = 0.0f32;
                for si in 0..=ti {
                    dot += p_all[po + si] * dp[si];
                }
                let mut ds = vec![0.0f32; t];
                for si in 0..=ti {
                    ds[si] = p_all[po + si] * (dp[si] - dot);
                }

                // scores backward (with /sqrt(Dh)): dQ += scale * sum_s dS[s] * K[..,s]
                //                                    dK[..,s] += scale * dS[s] * Q[..,t]
                let qi = q_off(bi, ti, hi);
                for si in 0..=ti {
                    let ki = kv_off(bi, si, kh);
                    let dss = ds[si] * scale;
                    for d in 0..dh {
                        dq[qi + d] += dss * kr[ki + d];
                        dk[ki + d] += dss * qr[qi + d];
                    }
                }
            }
        }
    }

    (
        TensorValue::from_vec(q.shape.clone(), dq),
        TensorValue::from_vec(k.shape.clone(), dk),
        TensorValue::from_vec(v.shape.clone(), dv),
    )
}

struct GqaBackward {
    q: TensorValue,
    k: TensorValue,
    v: TensorValue,
    p_all: Vec<f32>,
    sh: GqaShape,
    q_target: GradTarget,
    k_target: GradTarget,
    v_target: GradTarget,
}

impl BackwardRecipe for GqaBackward {
    fn backward(&self, grad_outputs: &[TensorValue], _ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        let dout = &grad_outputs[0];
        let (dq, dk, dv) = raw_gqa_backward(dout, &self.q, &self.k, &self.v, &self.p_all, self.sh);
        vec![
            GradEdge {
                target: self.q_target.clone(),
                grad: dq,
            },
            GradEdge {
                target: self.k_target.clone(),
                grad: dk,
            },
            GradEdge {
                target: self.v_target.clone(),
                grad: dv,
            },
        ]
    }
}

/// GQA: causal attention with `Hq` query heads and `Hk` KV heads.
///
/// `q` is `[B,T,Hq,Dh]`, `k` and `v` are `[B,T,Hk,Dh]`, output is `[B,T,Hq,Dh]`.
pub fn gqa_attention(q: &Tensor, k: &Tensor, v: &Tensor, name: &str) -> Tensor {
    let sh = {
        let qv = q.inner.borrow();
        let kv = k.inner.borrow();
        let vv = v.inner.borrow();
        check_shape_qkv(&qv.value, &kv.value, &vv.value)
    };

    let (out_value, p_all) = {
        let qv = q.inner.borrow();
        let kv = k.inner.borrow();
        let vv = v.inner.borrow();
        raw_gqa_forward(&qv.value, &kv.value, &vv.value, sh)
    };

    let recipe: Option<Box<dyn BackwardRecipe>> =
        if crate::grad_mode::is_enabled_and_any_requires_grad(&[q, k, v]) {
            // Snapshot Q/K/V values for backward (cheap Rc clone of data buffers).
            let q_val = q.inner.borrow().value.clone();
            let k_val = k.inner.borrow().value.clone();
            let v_val = v.inner.borrow().value.clone();
            Some(Box::new(GqaBackward {
                q: q_val,
                k: k_val,
                v: v_val,
                p_all,
                sh,
                q_target: q.grad_target(),
                k_target: k.grad_target(),
                v_target: v.grad_target(),
            }))
        } else {
            None
        };

    crate::ops::finish_op(
        OpKind::GqaAttention,
        name,
        &[q, k, v],
        out_value,
        recipe,
        vec![],
    )
}
