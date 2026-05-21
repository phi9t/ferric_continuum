//! Embedding lookup: token IDs → dense vectors.
//!
//! Book reference: Ch.4 "Embedding and Unembedding",
//! <https://jax-ml.github.io/scaling-book/transformers/>
//!
//! **This is a lookup, NOT a matrix multiplication.**
//! `embedding(ids, weight)`: ids `[B,T]` (usize) + weight `[V,D]` → out `[B,T,D]`
//! by row-gathering — O(B·T·D) memory reads, zero multiply-accumulate FLOPs.
//!
//! The book's `6·B·T·D·V` dense-matmul formula applies to the **unembedding**
//! direction (projecting hidden states to logits), implemented in
//! `ops/loss.rs::cross_entropy`.  The forward embedding lookup has no
//! equivalent matmul cost.
//!
//! Backward: scatter-add into `dweight[V,D]` — O(B·T·D) scalar adds.

use crate::autograd::{BackwardCtx, BackwardRecipe, GradEdge, GradTarget, OpKind};
use crate::tensor::{Shape, Tensor, TensorValue};

// ---------------------------------------------------------------------------
// Embedding lookup: ids [B,T] (usize), weight [V,D] -> out [B,T,D]
// ---------------------------------------------------------------------------

fn raw_embedding_forward(ids: &[usize], weight: &TensorValue, b: usize, t: usize) -> TensorValue {
    let wd = &weight.shape.0;
    let (vocab, d) = (wd[0], wd[1]);
    let wr = weight.data.as_ref();

    let mut out = vec![0.0f32; b * t * d];
    for bi in 0..b {
        for ti in 0..t {
            let id = ids[bi * t + ti];
            assert!(id < vocab, "embedding: id {} >= vocab {}", id, vocab);
            for di in 0..d {
                out[bi * t * d + ti * d + di] = wr[id * d + di];
            }
        }
    }

    TensorValue::from_vec(Shape(vec![b, t, d]), out)
}

// dweight [V, D]  (gradient accumulated from all token positions)
fn raw_embedding_backward(
    dout: &TensorValue,
    ids: &[usize],
    vocab: usize,
    b: usize,
    t: usize,
    d: usize,
) -> TensorValue {
    let doutr = dout.data.as_ref();
    let mut dw = vec![0.0f32; vocab * d];

    for bi in 0..b {
        for ti in 0..t {
            let id = ids[bi * t + ti];
            for di in 0..d {
                dw[id * d + di] += doutr[bi * t * d + ti * d + di];
            }
        }
    }

    TensorValue::from_vec(Shape(vec![vocab, d]), dw)
}

struct EmbeddingBackward {
    ids: Vec<usize>,
    w_target: GradTarget,
    vocab: usize,
    b: usize,
    t: usize,
    d: usize,
}

impl BackwardRecipe for EmbeddingBackward {
    fn backward(&self, grad_outputs: &[TensorValue], _ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        let dout = &grad_outputs[0];
        let dw = raw_embedding_backward(dout, &self.ids, self.vocab, self.b, self.t, self.d);
        vec![GradEdge {
            target: self.w_target.clone(),
            grad: dw,
        }]
    }
}

pub fn embedding(ids: &[usize], b: usize, t: usize, weight: &Tensor, name: &str) -> Tensor {
    let wd = {
        let wv = weight.inner.borrow();
        let wdims = &wv.value.shape.0;
        assert_eq!(wdims.len(), 2, "embedding weight must be 2D [V,D]");
        (wdims[0], wdims[1])
    };
    let (vocab, d) = wd;

    let out_value = raw_embedding_forward(ids, &weight.inner.borrow().value, b, t);

    let recipe: Option<Box<dyn BackwardRecipe>> =
        if crate::grad_mode::is_enabled_and_any_requires_grad(&[weight]) {
            Some(Box::new(EmbeddingBackward {
                ids: ids.to_vec(),
                w_target: weight.grad_target(),
                vocab,
                b,
                t,
                d,
            }))
        } else {
            None
        };

    crate::ops::finish_op(
        OpKind::Embedding,
        name,
        &[weight],
        out_value,
        recipe,
        vec![],
    )
}
