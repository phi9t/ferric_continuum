//! DeepSeek V4.1 MoE math helpers.
//!
//! These are pure functions used by verifier fixtures. They intentionally do
//! not hold parameters, mutate caches, or depend on autograd `Tensor`.

use crate::tensor::{Shape, Tensor, TensorValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateShape {
    pub tokens: usize,
    pub dim: usize,
    pub experts: usize,
}

pub fn sqrtsoftplus_scores(
    x: &[f32],
    gate_weight: &[f32],
    gate_temp: f32,
    shape: GateShape,
) -> Vec<f32> {
    assert!(gate_temp > 0.0, "gate_temp must be positive");
    assert_eq!(
        x.len(),
        shape.tokens * shape.dim,
        "x length does not match shape"
    );
    assert_eq!(
        gate_weight.len(),
        shape.experts * shape.dim,
        "gate_weight length does not match shape"
    );

    // Upstream: `scores = linear(x.float(), weight.float()) / gate_temp; scores = sqrt(softplus(scores))`.
    let mut out = vec![0.0f32; shape.tokens * shape.experts];
    for t in 0..shape.tokens {
        for e in 0..shape.experts {
            let mut acc = 0.0f32;
            let w_base = e * shape.dim;
            let x_base = t * shape.dim;
            for d in 0..shape.dim {
                acc += x[x_base + d] * gate_weight[w_base + d];
            }
            let scaled = acc / gate_temp;
            out[t * shape.experts + e] = softplus(scaled).sqrt();
        }
    }
    out
}

pub fn select_experts(scores: &[f32], correction_bias: &[f32], topk: usize) -> Vec<usize> {
    select_experts_masked(scores, correction_bias, None, None, topk)
}

/// Per-token expert selection with an optional VL bias.
///
/// Mirrors upstream `Gate.forward`: the selection score is
/// `scores + where(image_mask, bias_vl, bias)`. When `bias_vl`/`image_mask` are
/// `None` every token uses `correction_bias`, matching [`select_experts`].
pub fn select_experts_masked(
    scores: &[f32],
    correction_bias: &[f32],
    bias_vl: Option<&[f32]>,
    image_mask: Option<&[bool]>,
    topk: usize,
) -> Vec<usize> {
    assert!(topk > 0, "topk must be positive");
    assert!(
        scores.len() % correction_bias.len() == 0,
        "scores length must be tokens * experts"
    );
    let experts = correction_bias.len();
    assert!(experts > 0, "experts must be positive");
    let tokens = scores.len() / experts;
    if let Some(vl) = bias_vl {
        assert_eq!(vl.len(), experts, "bias_vl length must equal experts");
    }
    if let Some(mask) = image_mask {
        assert_eq!(mask.len(), tokens, "image_mask length must equal tokens");
    }

    // Selection uses (scores + effective_bias).topk(...)[1]
    let mut out = vec![0usize; tokens * topk];
    for t in 0..tokens {
        let base = t * experts;
        let use_vl = matches!((image_mask, bias_vl), (Some(mask), Some(_)) if mask[t]);
        let bias = if use_vl {
            bias_vl.expect("bias_vl present when use_vl")
        } else {
            correction_bias
        };
        let mut candidates = (0..experts)
            .map(|e| (e, scores[base + e] + bias[e]))
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        for (slot, (idx, _)) in candidates.into_iter().take(topk).enumerate() {
            out[t * topk + slot] = idx;
        }
    }
    out
}

pub fn route_weights(
    scores: &[f32],
    indices: &[usize],
    norm_topk_prob: bool,
    route_scale: f32,
) -> Vec<f32> {
    assert!(!scores.is_empty(), "scores must be non-empty");
    assert!(!indices.is_empty(), "indices must be non-empty");
    assert!(route_scale.is_finite(), "route_scale must be finite");
    assert!(
        scores.iter().all(|v| v.is_finite()),
        "scores must be finite"
    );

    let experts = scores.len();
    let topk = indices.len();
    let mut out = vec![0.0f32; topk];
    for (slot, &expert) in indices.iter().enumerate() {
        assert!(expert < experts, "expert index out of bounds for scores");
        out[slot] = scores[expert];
    }

    if norm_topk_prob && topk > 1 {
        let denom = out.iter().sum::<f32>() + 1e-20;
        for w in &mut out {
            *w /= denom;
        }
    }

    for w in &mut out {
        *w *= route_scale;
    }
    out
}

pub fn expert_swiglu(gate: &[f32], up: &[f32], swiglu_limit: f32) -> Vec<f32> {
    assert_eq!(gate.len(), up.len(), "gate and up must have same length");
    let mut out = Vec::with_capacity(gate.len());

    for (&g, &u) in gate.iter().zip(up.iter()) {
        let mut gate_val = g;
        let mut up_val = u;
        if swiglu_limit > 0.0 {
            up_val = up_val.clamp(-swiglu_limit, swiglu_limit);
            gate_val = gate_val.min(swiglu_limit);
        }
        out.push(silu(gate_val) * up_val);
    }
    out
}

pub struct DeepSeekV41Gate {
    pub weight: Vec<f32>,
    pub correction_bias: Vec<f32>,
    /// Optional VL routing bias applied to image-span tokens (upstream `bias_vl`).
    pub bias_vl: Option<Vec<f32>>,
    pub tokens: usize,
    pub dim: usize,
    pub experts: usize,
    pub topk: usize,
    pub gate_temp: f32,
    pub norm_topk_prob: bool,
    pub route_scale: f32,
}

impl DeepSeekV41Gate {
    pub fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<usize>, Vec<f32>) {
        self.forward_masked(x, None)
    }

    /// Gate forward with an optional per-token image mask that switches the
    /// correction bias to `bias_vl`. Routing weights always come from the
    /// unbiased scores, so `image_mask` affects selection only.
    pub fn forward_masked(
        &self,
        x: &[f32],
        image_mask: Option<&[bool]>,
    ) -> (Vec<f32>, Vec<usize>, Vec<f32>) {
        let scores = sqrtsoftplus_scores(
            x,
            &self.weight,
            self.gate_temp,
            GateShape {
                tokens: self.tokens,
                dim: self.dim,
                experts: self.experts,
            },
        );
        let indices = select_experts_masked(
            &scores,
            &self.correction_bias,
            self.bias_vl.as_deref(),
            image_mask,
            self.topk,
        );
        let mut weights = Vec::with_capacity(self.tokens * self.topk);
        for token in 0..self.tokens {
            weights.extend(route_weights(
                &scores[token * self.experts..(token + 1) * self.experts],
                &indices[token * self.topk..(token + 1) * self.topk],
                self.norm_topk_prob,
                self.route_scale,
            ));
        }
        (scores, indices, weights)
    }
}

pub struct DeepSeekV41Expert {
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
    pub w3: Vec<f32>,
    pub dim: usize,
    pub inter_dim: usize,
    pub swiglu_limit: f32,
}

impl DeepSeekV41Expert {
    pub fn forward_row(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.dim, "expert input row dim mismatch");
        assert_eq!(
            self.w1.len(),
            self.dim * self.inter_dim,
            "w1 shape mismatch"
        );
        assert_eq!(
            self.w2.len(),
            self.inter_dim * self.dim,
            "w2 shape mismatch"
        );
        assert_eq!(
            self.w3.len(),
            self.dim * self.inter_dim,
            "w3 shape mismatch"
        );

        let gate = matmul_row(x, &self.w1, self.dim, self.inter_dim);
        let up = matmul_row(x, &self.w3, self.dim, self.inter_dim);
        let hidden = expert_swiglu(&gate, &up, self.swiglu_limit);
        matmul_row(&hidden, &self.w2, self.inter_dim, self.dim)
    }
}

pub struct DeepSeekV41MoE {
    pub gate: DeepSeekV41Gate,
    pub experts: Vec<DeepSeekV41Expert>,
    pub shared_experts: DeepSeekV41Expert,
}

impl DeepSeekV41MoE {
    pub fn last_selected_experts(&self, x: &Tensor) -> Vec<usize> {
        let xv = x.inner.borrow();
        let (tokens, dim) = flatten_tokens(&xv.value.shape.0);
        assert_eq!(tokens, self.gate.tokens, "gate token count mismatch");
        assert_eq!(dim, self.gate.dim, "gate dim mismatch");
        let (_, indices, _) = self.gate.forward(xv.value.data.as_ref());
        indices
    }

    pub fn forward_layer(&self, x: &Tensor) -> Tensor {
        self.forward_layer_masked(x, None)
    }

    /// MoE forward with an optional image mask threaded to the gate so image
    /// tokens select experts via `bias_vl`. Mirrors `MoE.forward(x, image_mask)`.
    pub fn forward_layer_masked(&self, x: &Tensor, image_mask: Option<&[bool]>) -> Tensor {
        let xv = x.inner.borrow().value.clone();
        let shape = xv.shape.0.clone();
        let (tokens, dim) = flatten_tokens(&shape);
        assert_eq!(tokens, self.gate.tokens, "gate token count mismatch");
        assert_eq!(dim, self.gate.dim, "gate dim mismatch");
        assert_eq!(
            self.experts.len(),
            self.gate.experts,
            "expert count mismatch"
        );

        let (_, indices, weights) = self.gate.forward_masked(xv.data.as_ref(), image_mask);
        let mut out = vec![0.0f32; tokens * dim];
        for token in 0..tokens {
            let row = &xv.data[token * dim..(token + 1) * dim];
            for top in 0..self.gate.topk {
                let expert_idx = indices[token * self.gate.topk + top];
                let weight = weights[token * self.gate.topk + top];
                let expert_out = self.experts[expert_idx].forward_row(row);
                for d in 0..dim {
                    out[token * dim + d] += weight * expert_out[d];
                }
            }
            let shared = self.shared_experts.forward_row(row);
            for d in 0..dim {
                out[token * dim + d] += shared[d];
            }
        }
        Tensor::from_value_no_grad(TensorValue::from_vec(Shape(shape), out))
    }
}

fn flatten_tokens(shape: &[usize]) -> (usize, usize) {
    assert!(shape.len() >= 2, "layer input must be at least 2D");
    let dim = *shape.last().expect("last dim");
    let tokens = shape[..shape.len() - 1].iter().product();
    (tokens, dim)
}

fn matmul_row(x: &[f32], w: &[f32], din: usize, dout: usize) -> Vec<f32> {
    assert_eq!(x.len(), din, "matmul row input mismatch");
    assert_eq!(w.len(), din * dout, "matmul row weight mismatch");
    let mut out = vec![0.0f32; dout];
    for j in 0..dout {
        let mut acc = 0.0f32;
        for i in 0..din {
            acc += x[i] * w[i * dout + j];
        }
        out[j] = acc;
    }
    out
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    // numerically stable softplus.
    if x > 20.0 {
        x
    } else if x < -20.0 {
        (x).exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}
