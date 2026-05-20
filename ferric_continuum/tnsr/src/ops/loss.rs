//! Softmax cross-entropy loss (unembedding direction).
//!
//! Book reference: Ch.4 "Embedding and Unembedding",
//! <https://jax-ml.github.io/scaling-book/transformers/>
//!
//! `cross_entropy(logits [B,T,V], targets [B·T]) → scalar`
//!
//! The logit computation `hidden [B,T,D] @ W_out [D,V] → logits [B,T,V]` is
//! the "unembedding" matmul; its FLOPs are 2·B·T·D·V fwd and 4·B·T·D·V bwd.
//! This function receives already-computed logits and applies fused
//! softmax + cross-entropy, costing O(B·T·V) arithmetic ops.

use crate::autograd::{BackwardCtx, BackwardRecipe, GradEdge, GradTarget, OpKind};
use crate::tensor::{Shape, Tensor, TensorValue};

// ---------------------------------------------------------------------------
// Cross-entropy: logits [B,T,V] or [N,V], targets [N] -> scalar
// loss = -mean_n log(softmax(logits_n)[target_n])
// ---------------------------------------------------------------------------

fn softmax_row(logits: &[f32]) -> Vec<f32> {
    let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max_v).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&v| v / sum).collect()
}

fn raw_cross_entropy_forward(logits: &TensorValue, targets: &[usize]) -> (TensorValue, Vec<f32>) {
    let ld = &logits.shape.0;
    let v = *ld.last().unwrap();
    let n: usize = ld[..ld.len() - 1].iter().product();
    assert_eq!(
        n,
        targets.len(),
        "cross_entropy: n={} but targets.len()={}",
        n,
        targets.len()
    );

    let lr = logits.data.as_ref();
    let mut loss_sum = 0.0f32;
    let mut softmax_cache: Vec<f32> = vec![0.0f32; n * v];

    for i in 0..n {
        let row = &lr[i * v..(i + 1) * v];
        let sm = softmax_row(row);
        let target = targets[i];
        loss_sum -= sm[target].max(1e-10).ln();
        softmax_cache[i * v..(i + 1) * v].copy_from_slice(&sm);
    }

    let loss = loss_sum / n as f32;
    (
        TensorValue::from_vec(Shape(vec![1]), vec![loss]),
        softmax_cache,
    )
}

fn raw_cross_entropy_backward(
    dloss: f32,
    softmax_cache: &[f32],
    targets: &[usize],
    logits_shape: &Shape,
) -> TensorValue {
    let ld = &logits_shape.0;
    let v = *ld.last().unwrap();
    let n: usize = ld[..ld.len() - 1].iter().product();
    let scale = dloss / n as f32;

    let mut dlogits = vec![0.0f32; logits_shape.numel()];
    for i in 0..n {
        for j in 0..v {
            dlogits[i * v + j] = scale * softmax_cache[i * v + j];
        }
        dlogits[i * v + targets[i]] -= scale;
    }

    TensorValue::from_vec(logits_shape.clone(), dlogits)
}

struct CrossEntropyBackward {
    softmax_cache: Vec<f32>,
    targets: Vec<usize>,
    logits_shape: Shape,
    logits_target: GradTarget,
}

impl BackwardRecipe for CrossEntropyBackward {
    fn backward(&self, grad_outputs: &[TensorValue], _ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        let dloss = grad_outputs[0].data[0];
        let dlogits = raw_cross_entropy_backward(
            dloss,
            &self.softmax_cache,
            &self.targets,
            &self.logits_shape,
        );
        vec![GradEdge {
            target: self.logits_target.clone(),
            grad: dlogits,
        }]
    }
}

pub fn cross_entropy(logits: &Tensor, targets: &[usize], name: &str) -> Tensor {
    let logits_shape = logits.inner.borrow().value.shape.clone();
    let (out_value, softmax_cache) =
        raw_cross_entropy_forward(&logits.inner.borrow().value, targets);

    let recipe: Option<Box<dyn BackwardRecipe>> =
        if crate::grad_mode::is_enabled_and_any_requires_grad(&[logits]) {
            Some(Box::new(CrossEntropyBackward {
                softmax_cache,
                targets: targets.to_vec(),
                logits_shape,
                logits_target: logits.grad_target(),
            }))
        } else {
            None
        };

    crate::ops::finish_op(
        OpKind::CrossEntropy,
        name,
        &[logits],
        out_value,
        recipe,
        vec![],
    )
}
