//! DeepSeek V4.1 text attention layer skeleton.
//!
//! This module owns the parameter-bearing attention structs used by the
//! layer/block verifier ladder. It intentionally implements a small CPU
//! prefill path for deterministic tests; checkpoint loading, quantized kernels,
//! and optimized decode are later tickets.

use crate::ops::{linear, norm, shape as shape_ops};
use crate::tensor::{Shape, Tensor, TensorValue};

use super::hc_tensor::to_flat;
use super::sparse::compress_ratio_one;

#[derive(Default, Debug, Clone)]
pub struct SharedAttentionState {
    pub compressed_kv: Option<Vec<f32>>,
    pub index_k: Option<Vec<f32>>,
    pub topk_indices: Option<Vec<usize>>,
    pub consumed_sparse_indices: bool,
}

pub struct AttentionLayerInput<'a> {
    pub x: &'a Tensor,
    pub start_pos: usize,
    pub shared: &'a mut SharedAttentionState,
}

pub struct DeepSeekV41Compressor {
    pub wkv: Tensor,
    pub norm: Tensor,
    pub eps: f32,
    pub compress_ratio: usize,
}

impl DeepSeekV41Compressor {
    pub fn ratio_one(wkv: Tensor, norm: Tensor, eps: f32) -> Self {
        Self {
            wkv,
            norm,
            eps,
            compress_ratio: 1,
        }
    }

    pub fn forward_ratio_one(&self, x: &Tensor) -> Tensor {
        assert_eq!(
            self.compress_ratio, 1,
            "only ratio-one compressor is implemented in the layer skeleton"
        );
        let kv = linear::linear(x, &self.wkv, "compressor.wkv");
        let data = to_flat(&kv);
        let norm_weight = to_flat(&self.norm);
        let out = compress_ratio_one(&data, &norm_weight, self.eps);
        Tensor::from_value_no_grad(TensorValue::from_vec(kv.shape(), out))
    }
}

pub struct DeepSeekV41Indexer {
    pub index_topk: usize,
}

pub struct DeepSeekV41Attention {
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub compress_ratio: usize,
    pub window_size: usize,
    pub rms_norm_eps: f32,

    pub wq_a: Tensor,
    pub q_norm: Tensor,
    pub wq_b: Tensor,
    pub wkv: Tensor,
    pub kv_norm: Tensor,
    pub wo_a: Tensor,
    pub wo_b: Tensor,
    pub attn_sink: Tensor,

    pub compressor: Option<DeepSeekV41Compressor>,
    pub indexer: Option<DeepSeekV41Indexer>,
}

impl DeepSeekV41Attention {
    pub fn forward_layer(&self, input: AttentionLayerInput<'_>) -> Tensor {
        let (b, s, _d) = shape3(input.x, "DeepSeekV41Attention input");
        assert!(self.n_heads > 0, "n_heads must be positive");
        assert!(self.head_dim > 0, "head_dim must be positive");
        assert!(
            self.rope_head_dim <= self.head_dim,
            "rope_head_dim must not exceed head_dim"
        );
        assert!(
            self.n_heads % self.o_groups == 0,
            "n_heads must be divisible by o_groups"
        );
        let _ = (self.q_lora_rank, self.window_size, self.rms_norm_eps);

        let qr = linear::linear(input.x, &self.wq_a, "deepseek.attn.wq_a");
        let qr = norm::rms_norm(&qr, &self.q_norm, "deepseek.attn.q_norm");
        let q_flat = linear::linear(&qr, &self.wq_b, "deepseek.attn.wq_b");

        let kv = linear::linear(input.x, &self.wkv, "deepseek.attn.wkv");
        let kv = norm::rms_norm(&kv, &self.kv_norm, "deepseek.attn.kv_norm");
        let kv_data = to_flat(&kv);

        if let Some(compressor) = &self.compressor {
            let compressed = compressor.forward_ratio_one(input.x);
            input.shared.compressed_kv = Some(to_flat(&compressed));
        }
        if self.indexer.is_some() && input.shared.topk_indices.is_none() {
            input.shared.topk_indices = Some(default_sparse_indices(b, s));
        }
        if input.shared.topk_indices.is_some() {
            input.shared.consumed_sparse_indices = true;
        }

        let q = to_flat(&q_flat);
        let topk = input.shared.topk_indices.as_deref();
        let attended = sparse_attention(
            &q,
            &kv_data,
            topk,
            b,
            s,
            self.n_heads,
            self.head_dim,
            self.attn_sink.inner.borrow().value.data.as_ref(),
        );
        let low_rank = grouped_wo_a(
            &attended,
            b,
            s,
            self.n_heads,
            self.head_dim,
            self.o_groups,
            self.o_lora_rank,
            self.wo_a.inner.borrow().value.data.as_ref(),
        );
        let low_rank_tensor = Tensor::from_value_no_grad(TensorValue::from_vec(
            Shape(vec![b, s, self.o_groups * self.o_lora_rank]),
            low_rank,
        ));
        shape_ops::reshape(
            &linear::linear(&low_rank_tensor, &self.wo_b, "deepseek.attn.wo_b"),
            &[b, s, self.wo_b.inner.borrow().value.shape.0[1]],
            "deepseek.attn.out",
        )
    }
}

fn shape3(x: &Tensor, label: &str) -> (usize, usize, usize) {
    let shape = x.shape().0;
    assert_eq!(shape.len(), 3, "{label} must be [B,S,D]");
    (shape[0], shape[1], shape[2])
}

fn sparse_attention(
    q: &[f32],
    kv: &[f32],
    topk_indices: Option<&[usize]>,
    batch: usize,
    seqlen: usize,
    n_heads: usize,
    head_dim: usize,
    attn_sink: &[f32],
) -> Vec<f32> {
    assert_eq!(q.len(), batch * seqlen * n_heads * head_dim);
    assert_eq!(kv.len(), batch * seqlen * head_dim);
    assert_eq!(attn_sink.len(), n_heads);
    let topk = topk_indices.map(|idx| {
        assert_eq!(
            idx.len(),
            batch * seqlen,
            "fixture layer expects one sparse index per query"
        );
        idx
    });
    let scale = (head_dim as f32).powf(-0.5);
    let mut out = vec![0.0f32; batch * seqlen * n_heads * head_dim];

    for bi in 0..batch {
        for qi in 0..seqlen {
            for h in 0..n_heads {
                let src = topk
                    .map(|idx| idx[bi * seqlen + qi])
                    .unwrap_or_else(|| qi.saturating_sub(1).min(qi));
                assert!(src < seqlen, "sparse index out of bounds");
                let mut score = attn_sink[h];
                for d in 0..head_dim {
                    let q_idx = ((bi * seqlen + qi) * n_heads + h) * head_dim + d;
                    let k_idx = (bi * seqlen + src) * head_dim + d;
                    score += q[q_idx] * kv[k_idx] * scale;
                }
                let gate = 1.0 / (1.0 + (-score).exp());
                for d in 0..head_dim {
                    let out_idx = ((bi * seqlen + qi) * n_heads + h) * head_dim + d;
                    let v_idx = (bi * seqlen + src) * head_dim + d;
                    out[out_idx] = gate * kv[v_idx];
                }
            }
        }
    }
    out
}

fn default_sparse_indices(batch: usize, seqlen: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(batch * seqlen);
    for _ in 0..batch {
        for qi in 0..seqlen {
            out.push(qi.saturating_sub(1).min(qi));
        }
    }
    out
}

fn grouped_wo_a(
    attended: &[f32],
    batch: usize,
    seqlen: usize,
    n_heads: usize,
    head_dim: usize,
    o_groups: usize,
    o_lora_rank: usize,
    wo_a: &[f32],
) -> Vec<f32> {
    let heads_per_group = n_heads / o_groups;
    let group_in = heads_per_group * head_dim;
    assert_eq!(wo_a.len(), o_groups * group_in * o_lora_rank);
    let mut out = vec![0.0f32; batch * seqlen * o_groups * o_lora_rank];
    for b in 0..batch {
        for s in 0..seqlen {
            for g in 0..o_groups {
                for r in 0..o_lora_rank {
                    let mut acc = 0.0f32;
                    for i in 0..group_in {
                        let h = g * heads_per_group + i / head_dim;
                        let d = i % head_dim;
                        let a_idx = ((b * seqlen + s) * n_heads + h) * head_dim + d;
                        let w_idx = (g * group_in + i) * o_lora_rank + r;
                        acc += attended[a_idx] * wo_a[w_idx];
                    }
                    out[((b * seqlen + s) * o_groups + g) * o_lora_rank + r] = acc;
                }
            }
        }
    }
    out
}
