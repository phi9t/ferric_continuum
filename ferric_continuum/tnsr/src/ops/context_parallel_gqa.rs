//! Logical context-parallel grouped-query attention.
//!
//! This module makes the context-parallel data flow explicit while staying
//! inside `tnsr`'s single-process CPU model. Each logical rank owns an equal,
//! contiguous sequence shard. Forward materializes batch-major global K/V,
//! computes only that rank's query rows, and retains one probability block per
//! rank for the explicit backward pass.

use crate::autograd::{BackwardCtx, BackwardRecipe, GradEdge, GradTarget, OpKind};
use crate::tensor::{Shape, Tensor, TensorValue};

/// Validated dimensions shared by context-parallel forward and backward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextParallelGqaShape {
    pub b: usize,
    pub cp: usize,
    pub local_t: usize,
    pub global_t: usize,
    pub hq: usize,
    pub hk: usize,
    pub dh: usize,
}

impl ContextParallelGqaShape {
    fn group_size(self) -> usize {
        self.hq / self.hk
    }
}

/// Forward state needed to apply the exact local chain rule in backward.
pub struct ContextParallelGqaSaved {
    pub shape: ContextParallelGqaShape,
    // Rank r stores [B,Hq,local_t,global_t] in row-major order.
    probabilities: Vec<Vec<f32>>,
}

/// Explicit gradients returned to the three logical-shard input groups.
pub struct ContextParallelGqaGrads {
    pub dq: Vec<TensorValue>,
    pub dk: Vec<TensorValue>,
    pub dv: Vec<TensorValue>,
}

fn validate_shards(
    q_shards: &[TensorValue],
    k_shards: &[TensorValue],
    v_shards: &[TensorValue],
) -> ContextParallelGqaShape {
    assert!(
        !q_shards.is_empty(),
        "context_parallel_gqa: at least one shard is required"
    );
    assert_eq!(
        k_shards.len(),
        q_shards.len(),
        "context_parallel_gqa: Q and K shard counts must match"
    );
    assert_eq!(
        v_shards.len(),
        q_shards.len(),
        "context_parallel_gqa: Q and V shard counts must match"
    );

    let q0 = &q_shards[0].shape.0;
    let k0 = &k_shards[0].shape.0;
    let v0 = &v_shards[0].shape.0;
    assert_eq!(q0.len(), 4, "context_parallel_gqa: Q must be [B,S,Hq,Dh]");
    assert_eq!(k0.len(), 4, "context_parallel_gqa: K must be [B,S,Hk,Dh]");
    assert_eq!(v0.len(), 4, "context_parallel_gqa: V must be [B,S,Hk,Dh]");

    let (b, local_t, hq, dh) = (q0[0], q0[1], q0[2], q0[3]);
    let hk = k0[2];
    assert!(
        local_t > 0,
        "context_parallel_gqa: local sequence length must be positive"
    );
    assert!(hk > 0, "context_parallel_gqa: Hk must be positive");
    assert_eq!(
        hq % hk,
        0,
        "context_parallel_gqa: Hq must be divisible by Hk"
    );
    assert_eq!(
        (k0[0], k0[1], k0[3]),
        (b, local_t, dh),
        "context_parallel_gqa: K batch, sequence, and head dimensions must match Q"
    );
    assert_eq!(
        v0.as_slice(),
        [b, local_t, hk, dh],
        "context_parallel_gqa: V shape must match K"
    );

    for rank in 0..q_shards.len() {
        assert_eq!(
            q_shards[rank].shape.0.as_slice(),
            [b, local_t, hq, dh],
            "context_parallel_gqa: every Q shard must have the same shape"
        );
        assert_eq!(
            k_shards[rank].shape.0.as_slice(),
            [b, local_t, hk, dh],
            "context_parallel_gqa: every K shard must have the same shape"
        );
        assert_eq!(
            v_shards[rank].shape.0.as_slice(),
            [b, local_t, hk, dh],
            "context_parallel_gqa: every V shard must have the same shape"
        );
    }

    ContextParallelGqaShape {
        b,
        cp: q_shards.len(),
        local_t,
        global_t: local_t * q_shards.len(),
        hq,
        hk,
        dh,
    }
}

/// Gather equal sequence shards into one batch-major `[B,T,H,Dh]` buffer.
fn gather_sequence(
    shards: &[TensorValue],
    b: usize,
    local_t: usize,
    h: usize,
    dh: usize,
) -> Vec<f32> {
    let global_t = local_t * shards.len();
    let mut full = vec![0.0f32; b * global_t * h * dh];
    for (rank, shard) in shards.iter().enumerate() {
        let src = shard.data.as_ref();
        for bi in 0..b {
            for local_ti in 0..local_t {
                let global_ti = rank * local_t + local_ti;
                for hi in 0..h {
                    let src_base = ((bi * local_t + local_ti) * h + hi) * dh;
                    let dst_base = ((bi * global_t + global_ti) * h + hi) * dh;
                    full[dst_base..dst_base + dh].copy_from_slice(&src[src_base..src_base + dh]);
                }
            }
        }
    }
    full
}

/// Scatter one batch-major `[B,T,H,Dh]` buffer into equal sequence shards.
fn scatter_sequence(
    full: &[f32],
    b: usize,
    cp: usize,
    local_t: usize,
    h: usize,
    dh: usize,
) -> Vec<TensorValue> {
    let global_t = cp * local_t;
    assert_eq!(full.len(), b * global_t * h * dh);
    (0..cp)
        .map(|rank| {
            let mut shard = vec![0.0f32; b * local_t * h * dh];
            for bi in 0..b {
                for local_ti in 0..local_t {
                    let global_ti = rank * local_t + local_ti;
                    for hi in 0..h {
                        let src_base = ((bi * global_t + global_ti) * h + hi) * dh;
                        let dst_base = ((bi * local_t + local_ti) * h + hi) * dh;
                        shard[dst_base..dst_base + dh]
                            .copy_from_slice(&full[src_base..src_base + dh]);
                    }
                }
            }
            TensorValue::from_vec(Shape(vec![b, local_t, h, dh]), shard)
        })
        .collect()
}

/// Context-parallel causal GQA forward over equal contiguous logical shards.
///
/// Each output has shape `[B,S,Hq,Dh]`. Rank `r` owns global query positions
/// `[r*S,(r+1)*S)`, but consumes every causally visible key/value position from
/// the materialized global K/V buffers.
pub fn raw_context_parallel_gqa_forward(
    q_shards: &[TensorValue],
    k_shards: &[TensorValue],
    v_shards: &[TensorValue],
) -> (Vec<TensorValue>, ContextParallelGqaSaved) {
    let shape = validate_shards(q_shards, k_shards, v_shards);
    let ContextParallelGqaShape {
        b,
        cp,
        local_t,
        global_t,
        hq,
        hk,
        dh,
    } = shape;
    let group_size = shape.group_size();
    let scale = (dh as f32).sqrt().recip();
    let global_k = gather_sequence(k_shards, b, local_t, hk, dh);
    let global_v = gather_sequence(v_shards, b, local_t, hk, dh);

    let mut outputs = Vec::with_capacity(cp);
    let mut probabilities = Vec::with_capacity(cp);
    for (rank, q_shard) in q_shards.iter().enumerate() {
        let q = q_shard.data.as_ref();
        let mut out = vec![0.0f32; b * local_t * hq * dh];
        let mut p_rank = vec![0.0f32; b * hq * local_t * global_t];

        for bi in 0..b {
            for hi in 0..hq {
                let kh = hi / group_size;
                for local_q in 0..local_t {
                    let global_q = rank * local_t + local_q;
                    let q_base = ((bi * local_t + local_q) * hq + hi) * dh;
                    let p_base = ((bi * hq + hi) * local_t + local_q) * global_t;

                    let mut scores = vec![0.0f32; global_q + 1];
                    let mut row_max = f32::NEG_INFINITY;
                    for global_k_pos in 0..=global_q {
                        let k_base = ((bi * global_t + global_k_pos) * hk + kh) * dh;
                        let mut dot = 0.0f32;
                        for di in 0..dh {
                            dot += q[q_base + di] * global_k[k_base + di];
                        }
                        let score = dot * scale;
                        scores[global_k_pos] = score;
                        row_max = row_max.max(score);
                    }

                    let mut row_sum = 0.0f32;
                    for global_k_pos in 0..=global_q {
                        let probability = (scores[global_k_pos] - row_max).exp();
                        p_rank[p_base + global_k_pos] = probability;
                        row_sum += probability;
                    }
                    let inv_sum = row_sum.recip();
                    for global_k_pos in 0..=global_q {
                        p_rank[p_base + global_k_pos] *= inv_sum;
                    }

                    for di in 0..dh {
                        let mut value = 0.0f32;
                        for global_k_pos in 0..=global_q {
                            let v_base = ((bi * global_t + global_k_pos) * hk + kh) * dh;
                            value += p_rank[p_base + global_k_pos] * global_v[v_base + di];
                        }
                        out[q_base + di] = value;
                    }
                }
            }
        }

        outputs.push(TensorValue::from_vec(Shape(vec![b, local_t, hq, dh]), out));
        probabilities.push(p_rank);
    }

    (
        outputs,
        ContextParallelGqaSaved {
            shape,
            probabilities,
        },
    )
}

/// Explicit context-parallel GQA backward.
///
/// `dQ` is accumulated directly in each query owner's local buffer. Since a KV
/// position contributes to query rows on several logical ranks, `dK` and `dV`
/// first accumulate in batch-major global buffers and are then scattered back
/// to the rank that owns each contiguous sequence interval.
pub fn raw_context_parallel_gqa_backward(
    dout_shards: &[TensorValue],
    q_shards: &[TensorValue],
    k_shards: &[TensorValue],
    v_shards: &[TensorValue],
    saved: &ContextParallelGqaSaved,
) -> ContextParallelGqaGrads {
    let shape = validate_shards(q_shards, k_shards, v_shards);
    assert_eq!(
        saved.shape, shape,
        "context_parallel_gqa backward: saved shape does not match inputs"
    );
    assert_eq!(
        dout_shards.len(),
        shape.cp,
        "context_parallel_gqa backward: one output gradient is required per shard"
    );
    assert_eq!(saved.probabilities.len(), shape.cp);

    let ContextParallelGqaShape {
        b,
        cp,
        local_t,
        global_t,
        hq,
        hk,
        dh,
    } = shape;
    for dout in dout_shards {
        assert_eq!(
            dout.shape.0.as_slice(),
            [b, local_t, hq, dh],
            "context_parallel_gqa backward: dO shape must match each local output"
        );
    }

    let group_size = shape.group_size();
    let scale = (dh as f32).sqrt().recip();
    let global_k = gather_sequence(k_shards, b, local_t, hk, dh);
    let global_v = gather_sequence(v_shards, b, local_t, hk, dh);
    let mut dq_shards = Vec::with_capacity(cp);
    let mut global_dk = vec![0.0f32; b * global_t * hk * dh];
    let mut global_dv = vec![0.0f32; b * global_t * hk * dh];

    for rank in 0..cp {
        let q = q_shards[rank].data.as_ref();
        let dout = dout_shards[rank].data.as_ref();
        let probabilities = &saved.probabilities[rank];
        assert_eq!(probabilities.len(), b * hq * local_t * global_t);
        let mut dq = vec![0.0f32; b * local_t * hq * dh];

        for bi in 0..b {
            for hi in 0..hq {
                let kh = hi / group_size;
                for local_q in 0..local_t {
                    let global_q = rank * local_t + local_q;
                    let q_base = ((bi * local_t + local_q) * hq + hi) * dh;
                    let p_base = ((bi * hq + hi) * local_t + local_q) * global_t;

                    // dP[s] = dot(dO, V[s]); dV[s] += P[s] * dO.
                    let mut dp = vec![0.0f32; global_q + 1];
                    for global_k_pos in 0..=global_q {
                        let kv_base = ((bi * global_t + global_k_pos) * hk + kh) * dh;
                        let probability = probabilities[p_base + global_k_pos];
                        let mut dp_value = 0.0f32;
                        for di in 0..dh {
                            dp_value += dout[q_base + di] * global_v[kv_base + di];
                            global_dv[kv_base + di] += probability * dout[q_base + di];
                        }
                        dp[global_k_pos] = dp_value;
                    }

                    // Softmax Jacobian-vector product:
                    // dS[i] = P[i] * (dP[i] - sum_j P[j] * dP[j]).
                    let mut probability_dot = 0.0f32;
                    for global_k_pos in 0..=global_q {
                        probability_dot += probabilities[p_base + global_k_pos] * dp[global_k_pos];
                    }

                    // Scores backward routes dQ locally and accumulates every
                    // cross-rank contribution to the global dK owner buffer.
                    for global_k_pos in 0..=global_q {
                        let kv_base = ((bi * global_t + global_k_pos) * hk + kh) * dh;
                        let ds = probabilities[p_base + global_k_pos]
                            * (dp[global_k_pos] - probability_dot)
                            * scale;
                        for di in 0..dh {
                            dq[q_base + di] += ds * global_k[kv_base + di];
                            global_dk[kv_base + di] += ds * q[q_base + di];
                        }
                    }
                }
            }
        }

        dq_shards.push(TensorValue::from_vec(Shape(vec![b, local_t, hq, dh]), dq));
    }

    ContextParallelGqaGrads {
        dq: dq_shards,
        dk: scatter_sequence(&global_dk, b, cp, local_t, hk, dh),
        dv: scatter_sequence(&global_dv, b, cp, local_t, hk, dh),
    }
}

struct ContextParallelGqaBackward {
    q_shards: Vec<TensorValue>,
    k_shards: Vec<TensorValue>,
    v_shards: Vec<TensorValue>,
    saved: ContextParallelGqaSaved,
    q_targets: Vec<GradTarget>,
    k_targets: Vec<GradTarget>,
    v_targets: Vec<GradTarget>,
}

impl BackwardRecipe for ContextParallelGqaBackward {
    fn backward(&self, grad_outputs: &[TensorValue], _ctx: &mut BackwardCtx) -> Vec<GradEdge> {
        let grads = raw_context_parallel_gqa_backward(
            grad_outputs,
            &self.q_shards,
            &self.k_shards,
            &self.v_shards,
            &self.saved,
        );
        let mut edges = Vec::with_capacity(self.saved.shape.cp * 3);
        edges.extend(
            self.q_targets
                .iter()
                .cloned()
                .zip(grads.dq)
                .map(|(target, grad)| GradEdge { target, grad }),
        );
        edges.extend(
            self.k_targets
                .iter()
                .cloned()
                .zip(grads.dk)
                .map(|(target, grad)| GradEdge { target, grad }),
        );
        edges.extend(
            self.v_targets
                .iter()
                .cloned()
                .zip(grads.dv)
                .map(|(target, grad)| GradEdge { target, grad }),
        );
        edges
    }
}

/// Autograd-capable context-parallel GQA over explicit logical-rank tensors.
///
/// Input and gradient-edge order is fixed as all Q shards, all K shards, then
/// all V shards. The returned vector contains one output per logical rank, all
/// produced by a single multi-output autograd node.
pub fn context_parallel_gqa_attention(
    q_shards: &[Tensor],
    k_shards: &[Tensor],
    v_shards: &[Tensor],
    name: &str,
) -> Vec<Tensor> {
    let q_values: Vec<TensorValue> = q_shards
        .iter()
        .map(|t| t.inner.borrow().value.clone())
        .collect();
    let k_values: Vec<TensorValue> = k_shards
        .iter()
        .map(|t| t.inner.borrow().value.clone())
        .collect();
    let v_values: Vec<TensorValue> = v_shards
        .iter()
        .map(|t| t.inner.borrow().value.clone())
        .collect();
    let (outputs, saved) = raw_context_parallel_gqa_forward(&q_values, &k_values, &v_values);

    let mut inputs: Vec<&Tensor> = Vec::with_capacity(q_shards.len() * 3);
    inputs.extend(q_shards.iter());
    inputs.extend(k_shards.iter());
    inputs.extend(v_shards.iter());
    let recipe: Option<Box<dyn BackwardRecipe>> =
        if crate::grad_mode::is_enabled_and_any_requires_grad(&inputs) {
            Some(Box::new(ContextParallelGqaBackward {
                q_shards: q_values,
                k_shards: k_values,
                v_shards: v_values,
                saved,
                q_targets: q_shards.iter().map(Tensor::grad_target).collect(),
                k_targets: k_shards.iter().map(Tensor::grad_target).collect(),
                v_targets: v_shards.iter().map(Tensor::grad_target).collect(),
            }))
        } else {
            None
        };

    crate::ops::finish_op_multi(
        OpKind::ContextParallelGqaAttention,
        name,
        &inputs,
        outputs,
        recipe,
        vec![],
    )
}
