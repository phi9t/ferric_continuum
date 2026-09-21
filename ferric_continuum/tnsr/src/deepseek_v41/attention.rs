//! DeepSeek V4.1 text attention layer skeleton.
//!
//! This module owns the parameter-bearing attention structs used by the
//! layer/block verifier ladder. It intentionally implements a small CPU
//! prefill path for deterministic tests; checkpoint loading, quantized kernels,
//! and optimized decode are later tickets.

use std::collections::BTreeMap;

use crate::ops::{linear, norm, shape as shape_ops};
use crate::tensor::{Shape, Tensor, TensorValue};

use super::hc_tensor::to_flat;
use super::sparse::{
    compress_ratio_n, compress_ratio_one, select_candidate_blocks, window_topk_indices,
    CandidateLens, CandidateShape,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Csa2Mode {
    SlidingWindow,
    Full,
    Reindex,
    Reuse,
}

#[derive(Default, Clone)]
pub struct SharedAttentionState {
    pub compressed_kv: Option<Vec<f32>>,
    pub decoder_encoder_hidden: Option<Tensor>,
    pub index_k: Option<Vec<f32>>,
    pub topk_indices: Option<Vec<usize>>,
    pub candidate_mask: Option<Vec<bool>>,
    pub csa2_sources: BTreeMap<usize, Csa2SourceState>,
    pub swa_cache: Option<SwaCacheState>,
    pub consumed_decoder_encoder_hidden: bool,
    pub consumed_sparse_indices: bool,
    pub csa2_full_updates: usize,
    pub csa2_reindex_updates: usize,
    pub csa2_reuse_reads: usize,
}

#[derive(Default, Clone)]
pub struct Csa2SourceState {
    pub compressed_kv: Option<Vec<f32>>,
    pub index_k: Option<Vec<f32>>,
    pub topk_indices: Option<Vec<usize>>,
    pub candidate_mask: Option<Vec<bool>>,
}

#[derive(Default, Clone)]
pub struct SwaCacheState {
    pub batch: usize,
    pub window_size: usize,
    pub head_dim: usize,
    pub data: Vec<f32>,
}

pub struct AttentionLayerInput<'a> {
    pub x: &'a Tensor,
    pub start_pos: usize,
    pub shared: &'a mut SharedAttentionState,
}

pub struct DeepSeekV41Compressor {
    pub wkv: Tensor,
    pub wgate: Option<Tensor>,
    pub norm: Tensor,
    pub eps: f32,
    pub compress_ratio: usize,
}

impl DeepSeekV41Compressor {
    pub fn ratio_one(wkv: Tensor, norm: Tensor, eps: f32) -> Self {
        Self {
            wkv,
            wgate: None,
            norm,
            eps,
            compress_ratio: 1,
        }
    }

    pub fn ratio_n(
        wkv: Tensor,
        wgate: Tensor,
        norm: Tensor,
        eps: f32,
        compress_ratio: usize,
    ) -> Self {
        assert!(compress_ratio > 1, "ratio-N compressor requires ratio > 1");
        Self {
            wkv,
            wgate: Some(wgate),
            norm,
            eps,
            compress_ratio,
        }
    }

    pub fn forward(&self, x: &Tensor, start_pos: usize) -> Option<Tensor> {
        let kv = linear::linear(x, &self.wkv, "compressor.wkv");
        let norm_weight = to_flat(&self.norm);
        if self.compress_ratio == 1 {
            let out = compress_ratio_one(&to_flat(&kv), &norm_weight, self.eps);
            return Some(Tensor::from_value_no_grad(TensorValue::from_vec(
                kv.shape(),
                out,
            )));
        }

        assert_eq!(
            start_pos, 0,
            "ratio-N CSA2 decode compression state is not implemented yet"
        );
        let shape = kv.shape().0;
        assert_eq!(shape.len(), 3, "compressor.wkv output must be [B,T,D]");
        let batch = shape[0];
        let seqlen = shape[1];
        let head_dim = shape[2];
        let complete_groups = seqlen / self.compress_ratio;
        if complete_groups == 0 {
            return None;
        }

        let score = linear::linear(
            x,
            self.wgate
                .as_ref()
                .expect("ratio-N compressor requires attn.compressor.wgate.weight"),
            "compressor.wgate",
        );
        assert_eq!(
            score.shape().0,
            shape,
            "compressor.wgate output must match compressor.wkv output"
        );
        let kv_data = to_flat(&kv);
        let score_data = to_flat(&score);
        let complete_tokens = complete_groups * self.compress_ratio;
        let complete_len = batch * complete_tokens * head_dim;
        let mut complete_kv = Vec::with_capacity(complete_len);
        let mut complete_score = Vec::with_capacity(complete_len);
        for b in 0..batch {
            let batch_start = b * seqlen * head_dim;
            let batch_complete_end = batch_start + complete_tokens * head_dim;
            complete_kv.extend_from_slice(&kv_data[batch_start..batch_complete_end]);
            complete_score.extend_from_slice(&score_data[batch_start..batch_complete_end]);
        }
        let out = compress_ratio_n(
            &complete_kv,
            &complete_score,
            self.compress_ratio,
            &norm_weight,
            self.eps,
        );
        Some(Tensor::from_value_no_grad(TensorValue::from_vec(
            Shape(vec![batch, complete_groups, head_dim]),
            out,
        )))
    }
}

pub struct DeepSeekV41Indexer {
    pub index_topk: usize,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
    pub wq_b: Tensor,
    pub weights_proj: Tensor,
    pub wk: Option<Tensor>,
    pub k_norm: Option<Tensor>,
    pub eps: f32,
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

    pub layer_id: usize,
    pub kv_source_layer_id: Option<usize>,
    pub index_source_layer_id: Option<usize>,
    pub csa2_mode: Csa2Mode,
    pub compressor: Option<DeepSeekV41Compressor>,
    pub indexer: Option<DeepSeekV41Indexer>,
}

impl DeepSeekV41Attention {
    pub fn forward_layer(&self, input: AttentionLayerInput<'_>) -> Tensor {
        self.forward_layer_with_kv_source(input, None)
    }

    pub fn forward_layer_with_kv_source(
        &self,
        input: AttentionLayerInput<'_>,
        kv_prefix_source: Option<&Tensor>,
    ) -> Tensor {
        let (batch, query_tokens, _hidden_dim) = shape3(input.x, "DeepSeekV41Attention input");
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
        let _ = (self.q_lora_rank, self.rms_norm_eps);

        let qr = linear::linear(input.x, &self.wq_a, "deepseek.attn.wq_a");
        let qr = norm::rms_norm(&qr, &self.q_norm, "deepseek.attn.q_norm");
        let q_flat = linear::linear(&qr, &self.wq_b, "deepseek.attn.wq_b");

        let encoder_prefix = if kv_prefix_source.is_none() {
            input.shared.decoder_encoder_hidden.clone()
        } else {
            None
        };
        let kv_source = input.x;
        let (kv_b, _kv_s, _kv_d) = shape3(kv_source, "DeepSeekV41 attention KV source");
        assert_eq!(kv_b, batch, "attention KV source batch mismatch");
        if let Some(prefix) = encoder_prefix.as_ref() {
            let (prefix_b, _prefix_s, _prefix_d) =
                shape3(prefix, "DeepSeekV41 decoder encoder hidden");
            assert_eq!(prefix_b, batch, "decoder encoder hidden batch mismatch");
            input.shared.consumed_decoder_encoder_hidden = true;
        }

        // Build one flat KV coordinate system before attention:
        //
        //   `[batch, prefix_len + local_positions + compressed_positions, head_dim]`
        //
        // `topk` rows below are expressed in this final coordinate system. CED
        // and DSpark prefixes occupy the first positions, local SWA rows follow,
        // and compressed CSA2 rows are appended last.
        let local_kv = linear::linear(kv_source, &self.wkv, "deepseek.attn.wkv");
        let local_kv = norm::rms_norm(&local_kv, &self.kv_norm, "deepseek.attn.kv_norm");
        let local_kv_data = if kv_prefix_source.is_some() {
            to_flat(&local_kv)
        } else {
            windowed_local_kv(
                input.shared,
                &local_kv,
                batch,
                query_tokens,
                input.start_pos,
                self.window_size,
            )
        };
        let local_positions = local_kv_data.len() / (batch * self.head_dim);
        let mut kv_data = Vec::new();
        let mut prefix_len = if let Some(prefix) = kv_prefix_source {
            let (prefix_b, prefix_s, _prefix_d) = shape3(prefix, "DeepSeekV41 attention KV prefix");
            assert_eq!(prefix_b, batch, "attention KV prefix batch mismatch");
            let prefix_kv = linear::linear(prefix, &self.wkv, "deepseek.attn.prefix_wkv");
            let prefix_kv =
                norm::rms_norm(&prefix_kv, &self.kv_norm, "deepseek.attn.prefix_kv_norm");
            if input.start_pos == 0 {
                publish_swa_cache(
                    input.shared,
                    &to_flat(&prefix_kv),
                    batch,
                    prefix_s,
                    self.window_size,
                );
                0
            } else {
                kv_data.extend_from_slice(&to_flat(&prefix_kv));
                prefix_s
            }
        } else {
            0
        };
        let has_encoder_prefix = encoder_prefix.is_some();
        if let Some(prefix) = encoder_prefix.as_ref() {
            let (prefix_b, prefix_s, _prefix_d) =
                shape3(prefix, "DeepSeekV41 attention CED prefix");
            assert_eq!(prefix_b, batch, "attention CED prefix batch mismatch");
            let prefix_kv = linear::linear(prefix, &self.wkv, "deepseek.attn.ced_wkv");
            let prefix_kv = norm::rms_norm(&prefix_kv, &self.kv_norm, "deepseek.attn.ced_kv_norm");
            kv_data.extend_from_slice(&to_flat(&prefix_kv));
            prefix_len = prefix_s;
        }
        kv_data.extend_from_slice(&local_kv_data);

        let q = to_flat(&q_flat);
        self.update_csa2_state(
            input.x,
            &qr,
            batch,
            query_tokens,
            input.start_pos,
            local_positions,
            input.shared,
        );

        if let Some(compressed) = input.shared.compressed_kv.as_deref() {
            assert_eq!(
                compressed.len() % (batch * self.head_dim),
                0,
                "compressed KV length must be batch*positions*head_dim"
            );
            kv_data.extend_from_slice(compressed);
        }

        let generated_topk;
        let shifted_topk;
        let topk = if let Some(indices) = input.shared.topk_indices.as_deref() {
            input.shared.consumed_sparse_indices = true;
            if has_encoder_prefix {
                shifted_topk = with_prefix_global_topk(prefix_len, batch, query_tokens, indices);
                Some(shifted_topk.as_slice())
            } else {
                Some(indices)
            }
        } else if has_encoder_prefix {
            generated_topk = with_prefix_global_topk(
                prefix_len,
                batch,
                query_tokens,
                &window_indices_usize(self.window_size, batch, query_tokens, input.start_pos),
            );
            Some(generated_topk.as_slice())
        } else if self.window_size < query_tokens || input.start_pos > 0 {
            generated_topk =
                window_indices_usize(self.window_size, batch, query_tokens, input.start_pos);
            Some(generated_topk.as_slice())
        } else {
            None
        };
        let attended = sparse_attention(
            &q,
            &kv_data,
            topk,
            batch,
            query_tokens,
            prefix_len
                + local_positions
                + input
                    .shared
                    .compressed_kv
                    .as_ref()
                    .map_or(0, |kv| kv.len() / (batch * self.head_dim)),
            self.n_heads,
            self.head_dim,
            self.attn_sink.inner.borrow().value.data.as_ref(),
        );
        let low_rank = grouped_wo_a(
            &attended,
            batch,
            query_tokens,
            self.n_heads,
            self.head_dim,
            self.o_groups,
            self.o_lora_rank,
            self.wo_a.inner.borrow().value.data.as_ref(),
        );
        let low_rank_tensor = Tensor::from_value_no_grad(TensorValue::from_vec(
            Shape(vec![batch, query_tokens, self.o_groups * self.o_lora_rank]),
            low_rank,
        ));
        shape_ops::reshape(
            &linear::linear(&low_rank_tensor, &self.wo_b, "deepseek.attn.wo_b"),
            &[
                batch,
                query_tokens,
                self.wo_b.inner.borrow().value.shape.0[1],
            ],
            "deepseek.attn.out",
        )
    }

    fn update_csa2_state(
        &self,
        x: &Tensor,
        qr: &Tensor,
        batch: usize,
        seqlen: usize,
        start_pos: usize,
        local_positions: usize,
        shared: &mut SharedAttentionState,
    ) {
        match self.csa2_mode {
            Csa2Mode::SlidingWindow => {
                if let Some(compressor) = &self.compressor {
                    if let Some(compressed) = compressor.forward(x, start_pos) {
                        let compressed_kv = to_flat(&compressed);
                        shared.compressed_kv = Some(compressed_kv.clone());
                        shared.csa2_sources.insert(
                            self.kv_source_layer_id.unwrap_or(self.layer_id),
                            Csa2SourceState {
                                compressed_kv: Some(compressed_kv),
                                index_k: None,
                                topk_indices: None,
                                candidate_mask: None,
                            },
                        );
                    }
                }
                if self.indexer.is_some() && shared.topk_indices.is_none() {
                    shared.topk_indices = Some(default_sparse_indices(batch, seqlen));
                }
            }
            Csa2Mode::Full => {
                let compressor = self
                    .compressor
                    .as_ref()
                    .expect("CSA2 Full mode requires a compressor");
                let Some(compressed) = compressor.forward(x, start_pos) else {
                    return;
                };
                let compressed_shape = compressed.shape().0;
                let compressed_positions = compressed_shape[1];
                let source_id = self.kv_source_layer_id.unwrap_or(self.layer_id);
                let compressed_kv = to_flat(&compressed);
                shared.compressed_kv = Some(compressed_kv.clone());

                let indexer = self
                    .indexer
                    .as_ref()
                    .expect("CSA2 Full mode requires an indexer");
                let index_k = indexer_keys(indexer, &compressed);
                shared.index_k = Some(index_k.clone());
                let compressed_topk = select_index_topk(
                    indexer,
                    x,
                    qr,
                    &index_k,
                    IndexSelectionShape {
                        batch,
                        seqlen,
                        compressed_positions,
                        local_positions,
                        start_pos,
                        compress_ratio: self.compress_ratio.max(1),
                    },
                    shared,
                );
                let topk_indices = with_local_window_topk(
                    self.window_size,
                    batch,
                    seqlen,
                    start_pos,
                    &compressed_topk,
                );
                shared.topk_indices = Some(topk_indices.clone());
                shared.csa2_sources.insert(
                    source_id,
                    Csa2SourceState {
                        compressed_kv: Some(compressed_kv),
                        index_k: Some(index_k),
                        topk_indices: Some(topk_indices),
                        candidate_mask: shared.candidate_mask.clone(),
                    },
                );
                shared.csa2_full_updates += 1;
            }
            Csa2Mode::Reindex => {
                let indexer = self
                    .indexer
                    .as_ref()
                    .expect("CSA2 Reindex mode requires an indexer");
                let source_id = self
                    .kv_source_layer_id
                    .expect("CSA2 Reindex mode requires a configured KV source layer");
                let (compressed_kv, index_k) = {
                    let source = shared.csa2_sources.get(&source_id).unwrap_or_else(|| {
                        panic!("CSA2 Reindex mode requires source layer {source_id}")
                    });
                    let index_k = source
                        .index_k
                        .as_deref()
                        .expect("CSA2 Reindex mode requires cached index K")
                        .to_vec();
                    let compressed_kv = source
                        .compressed_kv
                        .as_deref()
                        .expect("CSA2 Reindex mode requires cached compressed KV")
                        .to_vec();
                    (compressed_kv, index_k)
                };
                shared.compressed_kv = Some(compressed_kv.clone());
                shared.index_k = Some(index_k.clone());
                let compressed_positions = compressed_kv.len() / (batch * self.head_dim);
                let compressed_topk = select_index_topk(
                    indexer,
                    x,
                    qr,
                    &index_k,
                    IndexSelectionShape {
                        batch,
                        seqlen,
                        compressed_positions,
                        local_positions,
                        start_pos,
                        compress_ratio: self.compress_ratio.max(1),
                    },
                    shared,
                );
                let topk_indices = with_local_window_topk(
                    self.window_size,
                    batch,
                    seqlen,
                    start_pos,
                    &compressed_topk,
                );
                shared.topk_indices = Some(topk_indices.clone());
                let index_source_id = self.index_source_layer_id.unwrap_or(self.layer_id);
                shared.csa2_sources.insert(
                    index_source_id,
                    Csa2SourceState {
                        compressed_kv: Some(compressed_kv),
                        index_k: Some(index_k),
                        topk_indices: Some(topk_indices),
                        candidate_mask: shared.candidate_mask.clone(),
                    },
                );
                shared.csa2_reindex_updates += 1;
            }
            Csa2Mode::Reuse => {
                let source_id = self
                    .index_source_layer_id
                    .expect("CSA2 Reuse mode requires a configured index source layer");
                let source = shared
                    .csa2_sources
                    .get(&source_id)
                    .unwrap_or_else(|| panic!("CSA2 Reuse mode requires source layer {source_id}"));
                shared.compressed_kv = Some(
                    source
                        .compressed_kv
                        .as_ref()
                        .expect("CSA2 Reuse mode requires cached compressed KV")
                        .clone(),
                );
                shared.topk_indices = Some(
                    source
                        .topk_indices
                        .as_ref()
                        .expect("CSA2 Reuse mode requires cached top-k indices")
                        .clone(),
                );
                shared.index_k = source.index_k.clone();
                shared.candidate_mask = source.candidate_mask.clone();
                shared.csa2_reuse_reads += 1;
            }
        }
    }
}

fn shape3(x: &Tensor, label: &str) -> (usize, usize, usize) {
    let shape = x.shape().0;
    assert_eq!(shape.len(), 3, "{label} must be [B,S,D]");
    (shape[0], shape[1], shape[2])
}

fn sparse_attention(
    query_by_token_head: &[f32],
    kv_by_token: &[f32],
    topk_indices: Option<&[usize]>,
    batch: usize,
    query_tokens: usize,
    kv_tokens: usize,
    heads: usize,
    head_dim: usize,
    attn_sink: &[f32],
) -> Vec<f32> {
    let layout = SparseAttentionLayout {
        batch,
        query_tokens,
        kv_tokens,
        heads,
        head_dim,
    };
    assert_eq!(query_by_token_head.len(), layout.query_numel());
    assert_eq!(kv_by_token.len(), layout.kv_numel());
    assert_eq!(attn_sink.len(), heads);
    let topk = topk_indices.map(|idx| {
        assert_eq!(
            idx.len() % layout.query_rows(),
            0,
            "sparse index count must be a multiple of batch*seqlen"
        );
        idx
    });
    let sparse_width = topk.map(|idx| idx.len() / layout.query_rows()).unwrap_or(1);
    let scale = (head_dim as f32).powf(-0.5);
    let mut out = vec![0.0f32; layout.query_numel()];

    for batch_index in 0..layout.batch {
        for query_position in 0..layout.query_tokens {
            for head in 0..layout.heads {
                let row = layout.row(batch_index, query_position);
                let fallback = query_position.saturating_sub(1).min(query_position);
                let candidates = topk
                    .map(|idx| &idx[row * sparse_width..(row + 1) * sparse_width])
                    .unwrap_or(std::slice::from_ref(&fallback));
                // Width 1 is the tracked sliding-window fixture path: the
                // single chosen KV row is gated by `attn_sink` with a sigmoid.
                if sparse_width == 1 {
                    let source_position = candidates[0];
                    if source_position == usize::MAX {
                        continue;
                    }
                    assert!(
                        source_position < layout.kv_tokens,
                        "sparse index out of bounds"
                    );
                    let mut score = attn_sink[head];
                    for head_feature in 0..layout.head_dim {
                        score += query_by_token_head
                            [layout.query_offset(batch_index, query_position, head, head_feature)]
                            * kv_by_token
                                [layout.kv_offset(batch_index, source_position, head_feature)]
                            * scale;
                    }
                    let gate = 1.0 / (1.0 + (-score).exp());
                    for head_feature in 0..layout.head_dim {
                        out[layout.query_offset(batch_index, query_position, head, head_feature)] =
                            gate * kv_by_token
                                [layout.kv_offset(batch_index, source_position, head_feature)];
                    }
                    continue;
                }
                // Wider sparse rows model CSA2: `usize::MAX` is the sink row,
                // real candidates compete by softmax, and only real KV rows
                // contribute values.
                let mut scores_by_source = Vec::with_capacity(candidates.len() + 1);
                scores_by_source.push((usize::MAX, attn_sink[head]));
                for &source_position in candidates {
                    if source_position == usize::MAX {
                        continue;
                    }
                    assert!(
                        source_position < layout.kv_tokens,
                        "sparse index out of bounds"
                    );
                    let mut score = 0.0;
                    for head_feature in 0..layout.head_dim {
                        score += query_by_token_head
                            [layout.query_offset(batch_index, query_position, head, head_feature)]
                            * kv_by_token
                                [layout.kv_offset(batch_index, source_position, head_feature)]
                            * scale;
                    }
                    scores_by_source.push((source_position, score));
                }
                let max_score = scores_by_source
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let denom: f32 = scores_by_source
                    .iter()
                    .map(|(_, score)| (*score - max_score).exp())
                    .sum();
                for &(source_position, score) in &scores_by_source {
                    if source_position == usize::MAX {
                        continue;
                    }
                    let weight = (score - max_score).exp() / denom;
                    for head_feature in 0..layout.head_dim {
                        out[layout.query_offset(
                            batch_index,
                            query_position,
                            head,
                            head_feature,
                        )] += weight
                            * kv_by_token
                                [layout.kv_offset(batch_index, source_position, head_feature)];
                    }
                }
            }
        }
    }
    out
}

/// Row-major layouts used by the educational sparse-attention skeleton.
///
/// - Query/output: `[batch, query_tokens, heads, head_dim]`.
/// - Key/value: `[batch, kv_tokens, head_dim]`; the same KV row is read by
///   every query head in this tiny CPU path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SparseAttentionLayout {
    batch: usize,
    query_tokens: usize,
    kv_tokens: usize,
    heads: usize,
    head_dim: usize,
}

impl SparseAttentionLayout {
    fn query_rows(self) -> usize {
        self.batch * self.query_tokens
    }

    fn query_numel(self) -> usize {
        self.query_rows() * self.heads * self.head_dim
    }

    fn kv_numel(self) -> usize {
        self.batch * self.kv_tokens * self.head_dim
    }

    fn row(self, batch_index: usize, query_position: usize) -> usize {
        batch_index * self.query_tokens + query_position
    }

    fn query_offset(
        self,
        batch_index: usize,
        query_position: usize,
        head: usize,
        head_feature: usize,
    ) -> usize {
        ((batch_index * self.query_tokens + query_position) * self.heads + head) * self.head_dim
            + head_feature
    }

    fn kv_offset(self, batch_index: usize, source_position: usize, head_feature: usize) -> usize {
        (batch_index * self.kv_tokens + source_position) * self.head_dim + head_feature
    }
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

fn windowed_local_kv(
    shared: &mut SharedAttentionState,
    local_kv: &Tensor,
    batch: usize,
    seqlen: usize,
    start_pos: usize,
    window_size: usize,
) -> Vec<f32> {
    let local = to_flat(local_kv);
    if start_pos == 0 {
        publish_swa_cache(shared, &local, batch, seqlen, window_size);
        return local;
    }

    assert_eq!(seqlen, 1, "decode attention expects one token per step");
    let cache = shared.swa_cache.get_or_insert_with(|| SwaCacheState {
        batch,
        window_size,
        head_dim: local.len() / batch,
        data: vec![0.0; batch * window_size * (local.len() / batch)],
    });
    assert_eq!(cache.batch, batch, "SWA cache batch mismatch");
    assert_eq!(cache.window_size, window_size, "SWA cache window mismatch");
    assert_eq!(
        local.len(),
        batch * cache.head_dim,
        "SWA decode KV shape mismatch"
    );
    let slot = start_pos % window_size;
    for b in 0..batch {
        let src = b * cache.head_dim..(b + 1) * cache.head_dim;
        let dst = (b * window_size + slot) * cache.head_dim
            ..(b * window_size + slot + 1) * cache.head_dim;
        cache.data[dst].copy_from_slice(&local[src]);
    }
    cache.data.clone()
}

fn publish_swa_cache(
    shared: &mut SharedAttentionState,
    local: &[f32],
    batch: usize,
    seqlen: usize,
    window_size: usize,
) {
    if window_size == 0 {
        return;
    }
    assert_eq!(local.len() % (batch * seqlen), 0, "local KV shape mismatch");
    let head_dim = local.len() / (batch * seqlen);
    let mut cache = SwaCacheState {
        batch,
        window_size,
        head_dim,
        data: vec![0.0; batch * window_size * head_dim],
    };
    let start = seqlen.saturating_sub(window_size);
    for b in 0..batch {
        for pos in start..seqlen {
            let slot = pos % window_size;
            let src = (b * seqlen + pos) * head_dim..(b * seqlen + pos + 1) * head_dim;
            let dst = (b * window_size + slot) * head_dim..(b * window_size + slot + 1) * head_dim;
            cache.data[dst].copy_from_slice(&local[src]);
        }
    }
    shared.swa_cache = Some(cache);
}

fn with_prefix_global_topk(
    prefix_len: usize,
    batch: usize,
    seqlen: usize,
    local_or_sparse_topk: &[usize],
) -> Vec<usize> {
    let rows = batch * seqlen;
    assert_eq!(
        local_or_sparse_topk.len() % rows,
        0,
        "top-k shape mismatch for prefix attention"
    );
    let width = local_or_sparse_topk.len() / rows;
    let mut out = Vec::with_capacity(rows * (prefix_len + width));
    for row in 0..rows {
        for prefix in 0..prefix_len {
            out.push(prefix);
        }
        out.extend(
            local_or_sparse_topk[row * width..(row + 1) * width]
                .iter()
                .map(|&idx| {
                    if idx == usize::MAX {
                        usize::MAX
                    } else {
                        idx + prefix_len
                    }
                }),
        );
    }
    out
}

fn window_indices_usize(
    window_size: usize,
    batch: usize,
    seqlen: usize,
    start_pos: usize,
) -> Vec<usize> {
    window_topk_indices(window_size, batch, seqlen, start_pos)
        .into_iter()
        .map(|idx| usize::try_from(idx).unwrap_or(usize::MAX))
        .collect()
}

struct IndexSelectionShape {
    batch: usize,
    seqlen: usize,
    compressed_positions: usize,
    local_positions: usize,
    start_pos: usize,
    compress_ratio: usize,
}

fn indexer_keys(indexer: &DeepSeekV41Indexer, compressed: &Tensor) -> Vec<f32> {
    let wk = indexer
        .wk
        .as_ref()
        .expect("CSA2 index-key owner requires indexer.wk");
    let k_norm = indexer
        .k_norm
        .as_ref()
        .expect("CSA2 index-key owner requires indexer.k_norm");
    let k = linear::linear(compressed, wk, "deepseek.attn.indexer.wk");
    to_flat(&norm::rms_norm(&k, k_norm, "deepseek.attn.indexer.k_norm"))
}

fn select_index_topk(
    indexer: &DeepSeekV41Indexer,
    x: &Tensor,
    qr: &Tensor,
    index_k: &[f32],
    shape: IndexSelectionShape,
    shared: &mut SharedAttentionState,
) -> Vec<usize> {
    let q_index = linear::linear(qr, &indexer.wq_b, "deepseek.attn.indexer.wq_b");
    let weights = linear::linear(
        x,
        &indexer.weights_proj,
        "deepseek.attn.indexer.weights_proj",
    );
    let q_shape = q_index.shape().0;
    let weights_shape = weights.shape().0;
    assert_eq!(q_shape.len(), 3, "indexer query must be [B,S,H*D]");
    assert_eq!(weights_shape.len(), 3, "indexer weights must be [B,S,H]");
    assert_eq!(q_shape[0], shape.batch, "indexer query batch mismatch");
    assert_eq!(q_shape[1], shape.seqlen, "indexer query sequence mismatch");
    assert_eq!(
        weights_shape[2] > 0,
        true,
        "indexer must have at least one head"
    );
    let index_heads = weights_shape[2];
    assert_eq!(
        q_shape[2] % index_heads,
        0,
        "indexer query dim must be divisible by index heads"
    );
    let index_head_dim = q_shape[2] / index_heads;
    assert_eq!(
        index_k.len(),
        shape.batch * shape.compressed_positions * index_head_dim,
        "cached index K shape mismatch"
    );
    let q = to_flat(&q_index);
    let weights = to_flat(&weights);
    let mut scores =
        vec![f32::NEG_INFINITY; shape.batch * shape.seqlen * shape.compressed_positions];

    for b in 0..shape.batch {
        for qi in 0..shape.seqlen {
            let row = b * shape.seqlen + qi;
            let visible = visible_compressed_positions(
                shape.start_pos,
                qi,
                shape.compress_ratio,
                shape.compressed_positions,
            );
            for pos in 0..visible {
                let mut score = 0.0f32;
                for h in 0..index_heads {
                    let mut dot = 0.0f32;
                    for d in 0..index_head_dim {
                        let q_idx =
                            ((b * shape.seqlen + qi) * index_heads + h) * index_head_dim + d;
                        let k_idx = (b * shape.compressed_positions + pos) * index_head_dim + d;
                        dot += q[q_idx] * index_k[k_idx];
                    }
                    let weight_idx = (b * shape.seqlen + qi) * index_heads + h;
                    score += dot.max(0.0) * weights[weight_idx];
                }
                scores[row * shape.compressed_positions + pos] =
                    score * (indexer.eps + (index_head_dim as f32).powf(-0.5));
            }
        }
    }

    let candidates = if indexer.candidate_topk_blocks > 0
        && indexer.candidate_block_size > 0
        && shape.compressed_positions > 0
    {
        let lens = (0..shape.batch * shape.seqlen)
            .map(|row| {
                let qi = row % shape.seqlen;
                visible_compressed_positions(
                    shape.start_pos,
                    qi,
                    shape.compress_ratio,
                    shape.compressed_positions,
                ) as i32
            })
            .collect();
        let mask = select_candidate_blocks(
            &scores,
            CandidateShape {
                batch: shape.batch,
                seqlen: shape.seqlen,
                positions: shape.compressed_positions,
            },
            CandidateLens::PerQuery(lens),
            indexer.candidate_topk_blocks,
            indexer.candidate_block_size,
        );
        shared.candidate_mask = Some(mask.clone());
        Some(mask)
    } else {
        None
    };

    let width = indexer.index_topk.min(shape.compressed_positions);
    let mut out = Vec::with_capacity(shape.batch * shape.seqlen * width);
    for b in 0..shape.batch {
        for qi in 0..shape.seqlen {
            let row = b * shape.seqlen + qi;
            let visible = visible_compressed_positions(
                shape.start_pos,
                qi,
                shape.compress_ratio,
                shape.compressed_positions,
            );
            let mut ranked = Vec::with_capacity(visible);
            for pos in 0..visible {
                if candidates
                    .as_ref()
                    .is_some_and(|mask| !mask[row * shape.compressed_positions + pos])
                {
                    continue;
                }
                ranked.push((pos, scores[row * shape.compressed_positions + pos]));
            }
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let mut row_indices = ranked
                .into_iter()
                .take(width)
                .map(|(pos, _)| pos + shape.local_positions)
                .collect::<Vec<_>>();
            row_indices.sort_unstable();
            row_indices.resize(width, usize::MAX);
            out.extend(row_indices);
        }
    }
    out
}

fn visible_compressed_positions(
    start_pos: usize,
    query_offset: usize,
    compress_ratio: usize,
    compressed_positions: usize,
) -> usize {
    ((start_pos + query_offset + 1) / compress_ratio).min(compressed_positions)
}

fn with_local_window_topk(
    window_size: usize,
    batch: usize,
    seqlen: usize,
    start_pos: usize,
    compressed_topk: &[usize],
) -> Vec<usize> {
    let local = window_indices_usize(window_size, batch, seqlen, start_pos);
    if compressed_topk.is_empty() {
        return local;
    }
    let rows = batch * seqlen;
    assert_eq!(local.len() % rows, 0, "local window top-k shape mismatch");
    assert_eq!(
        compressed_topk.len() % rows,
        0,
        "compressed top-k shape mismatch"
    );
    let local_width = local.len() / rows;
    let compressed_width = compressed_topk.len() / rows;
    let mut out = Vec::with_capacity(rows * (local_width + compressed_width));
    for row in 0..rows {
        out.extend_from_slice(&local[row * local_width..(row + 1) * local_width]);
        out.extend_from_slice(
            &compressed_topk[row * compressed_width..(row + 1) * compressed_width],
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_attention_layout_names_query_and_kv_offsets() {
        let layout = SparseAttentionLayout {
            batch: 2,
            query_tokens: 3,
            kv_tokens: 5,
            heads: 7,
            head_dim: 11,
        };

        assert_eq!(
            layout.query_offset(1, 2, 3, 4),
            ((1 * 3 + 2) * 7 + 3) * 11 + 4
        );
        assert_eq!(layout.kv_offset(1, 4, 10), (1 * 5 + 4) * 11 + 10);
        assert_eq!(layout.query_rows(), 6);
        assert_eq!(layout.query_numel(), 462);
        assert_eq!(layout.kv_numel(), 110);
    }

    #[test]
    fn with_prefix_global_topk_shifts_real_indices_and_preserves_sink() {
        let got = with_prefix_global_topk(2, 1, 2, &[usize::MAX, 1, 0, 1]);

        assert_eq!(got, vec![0, 1, usize::MAX, 3, 0, 1, 2, 3]);
    }

    #[test]
    fn with_local_window_topk_concatenates_rows_without_reordering() {
        let got = with_local_window_topk(2, 1, 3, 0, &[2, usize::MAX, 3]);

        assert_eq!(got, vec![0, usize::MAX, 2, 0, 1, usize::MAX, 1, 2, 3]);
    }

    #[test]
    fn visible_compressed_positions_keeps_incomplete_groups_invisible() {
        let visible = (0..5)
            .map(|query_offset| visible_compressed_positions(0, query_offset, 2, 4))
            .collect::<Vec<_>>();

        assert_eq!(visible, vec![0, 1, 1, 2, 2]);
        assert_eq!(visible_compressed_positions(8, 0, 2, 4), 4);
    }

    #[test]
    #[should_panic(expected = "sparse index count must be a multiple of batch*seqlen")]
    fn sparse_attention_rejects_topk_rows_not_divisible_by_batch_seqlen() {
        let _ = sparse_attention(
            &[1.0, 2.0],
            &[3.0, 4.0],
            Some(&[0, 1, 0]),
            1,
            2,
            2,
            1,
            1,
            &[0.0],
        );
    }

    #[test]
    #[should_panic(expected = "decode attention expects one token per step")]
    fn windowed_local_kv_rejects_multi_token_decode() {
        let mut shared = SharedAttentionState::default();
        let local_kv =
            Tensor::from_value_no_grad(TensorValue::from_vec(Shape(vec![1, 2, 1]), vec![1.0, 2.0]));

        let _ = windowed_local_kv(&mut shared, &local_kv, 1, 2, 3, 2);
    }
}
