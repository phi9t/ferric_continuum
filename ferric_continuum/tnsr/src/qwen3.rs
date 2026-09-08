//! Qwen3 (dense) architecture.
//!
//! Faithful port of the Qwen3 dense Transformer block as described in the
//! Qwen3 technical report (arXiv 2505.09388) and the Hugging Face
//! `transformers/models/qwen3` implementation.
//!
//! Differences from `transformer::TransformerBlock` (which models Qwen2's
//! ancestor with classic LayerNorm + GELU):
//!
//! 1. **RMSNorm** (γ only, no β) at pre-attention and pre-MLP positions.
//! 2. **GQA**: `n_q_heads` query heads + `n_kv_heads` shared K/V heads.
//! 3. **Bias-free** Q/K/V/O and gate/up/down projections.
//! 4. **Per-head Q/K RMSNorm** on the `[head_dim]` axis (gamma shared across
//!    heads), applied BEFORE RoPE.  This is the headline Qwen3 dense change.
//! 5. **RoPE** with configurable base (Qwen3 uses 1e6) on Q and K.
//! 6. **SwiGLU MLP**: `down_proj( silu(gate_proj(x)) * up_proj(x) )`.
//!
//! Full model: `embed → N × Qwen3Block → final RMSNorm → lm_head` (untied).
//!
//! Reference: HF `Qwen3Attention.forward` —
//! <https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py>

use crate::attention_layout::{AttentionLayoutError, EqualContiguousAttentionLayout};
use crate::ops::{
    activations, basic, context_parallel_gqa, embedding, gqa, linear, norm, rope,
    shape as shape_ops,
};
use crate::tensor::{Shape, Tensor, TensorValue};
use crate::typed::{
    AxisExtent, FullHiddenStates, HeadDim, HeadScale, Hidden, KvHead, KvProjectionWeight, Merged,
    OutputProjectionWeight, QueryHead, QueryProjectionWeight, ShardHiddenStates,
};

/// Hyperparameters for a Qwen3 dense model.
///
/// Defaults follow no specific checkpoint; pick a preset like
/// [`Qwen3Config::qwen3_8b`] or build your own.
#[derive(Clone, Debug)]
pub struct Qwen3Config {
    /// Vocabulary size for embeddings + LM head (HF embedding table may be
    /// padded; this is the logical token count).
    pub vocab_size: usize,
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Model hidden dim (a.k.a. `d_model`).
    pub hidden_size: usize,
    /// SwiGLU intermediate size (`d_ff`).
    pub intermediate_size: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of K/V heads (GQA).  Must divide `num_attention_heads`.
    pub num_key_value_heads: usize,
    /// Per-head dimension `Dh` (must be even for RoPE).
    pub head_dim: usize,
    /// RoPE base frequency (`rope_theta`).  Qwen3 = 1e6.
    pub rope_theta: f32,
    /// Use bias on Q/K/V/O projections.  False for standard Qwen3.
    pub attention_bias: bool,
    /// Tie LM head to embedding (small Qwen3 dense ties; large does not).
    pub tie_word_embeddings: bool,
}

impl Qwen3Config {
    /// Approximate Qwen3-8B (per the report / HF config).
    pub fn qwen3_8b() -> Self {
        Self {
            vocab_size: 151_936,
            num_hidden_layers: 36,
            hidden_size: 4096,
            intermediate_size: 12_288,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            attention_bias: false,
            tie_word_embeddings: false,
        }
    }

    /// Tiny config for unit tests: V=11, L=2, D=16, F=32, Hq=4, Hk=2, Dh=8.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 11,
            num_hidden_layers: 2,
            hidden_size: 16,
            intermediate_size: 32,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            rope_theta: 10_000.0,
            attention_bias: false,
            tie_word_embeddings: true,
        }
    }

    pub fn group_size(&self) -> usize {
        assert!(self.num_attention_heads % self.num_key_value_heads == 0);
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn q_total(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_total(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

fn ones(n: usize) -> Tensor {
    Tensor::from_value(TensorValue::from_vec(Shape(vec![n]), vec![1.0; n]), true)
}

fn param(rows: usize, cols: usize, scale: f32) -> Tensor {
    let t = Tensor::randn_scaled(&[rows, cols], scale);
    t.set_requires_grad(true);
    t
}

// ---------------------------------------------------------------------------
// Self-attention
// ---------------------------------------------------------------------------

pub struct Qwen3Attention {
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_cfg: rope::RopeConfig,

    pub wq: Tensor, // [D, Hq*Dh]
    pub wk: Tensor, // [D, Hk*Dh]
    pub wv: Tensor, // [D, Hk*Dh]
    pub wo: Tensor, // [Hq*Dh, D]

    pub q_norm: Tensor, // [Dh]
    pub k_norm: Tensor, // [Dh]
}

/// Named runtime geometry used by one typed attention call.
struct AttentionGeometry {
    hidden: AxisExtent<Hidden>,
    query_heads: AxisExtent<QueryHead>,
    kv_heads: AxisExtent<KvHead>,
    head_dim: AxisExtent<HeadDim>,
    query_width: AxisExtent<Merged<QueryHead, HeadDim>>,
    kv_width: AxisExtent<Merged<KvHead, HeadDim>>,
}

/// One coherent snapshot of Qwen attention's mutable public state.
///
/// Scalars are copied and tensor handles are cloned only for the duration of a
/// call. Parameter values are not copied. This is coherent because tnsr's
/// execution is single-threaded and callback-free.
struct ValidatedAttentionState {
    geometry: AttentionGeometry,
    rope_cfg: rope::RopeConfig,
    wq: QueryProjectionWeight,
    wk: KvProjectionWeight,
    wv: KvProjectionWeight,
    wo: OutputProjectionWeight,
    q_norm: HeadScale,
    k_norm: HeadScale,
}

impl Qwen3Attention {
    pub fn new(cfg: &Qwen3Config) -> Self {
        let d = cfg.hidden_size;
        let scale = (d as f32).sqrt().recip() * 0.5;
        Self {
            n_q_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rope_cfg: rope::RopeConfig {
                base: cfg.rope_theta,
                start_pos: 0,
            },
            wq: param(d, cfg.q_total(), scale),
            wk: param(d, cfg.kv_total(), scale),
            wv: param(d, cfg.kv_total(), scale),
            wo: param(cfg.q_total(), d, scale),
            q_norm: ones(cfg.head_dim),
            k_norm: ones(cfg.head_dim),
        }
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![
            &self.wq,
            &self.wk,
            &self.wv,
            &self.wo,
            &self.q_norm,
            &self.k_norm,
        ]
    }

    fn validated_attention_state(&self) -> ValidatedAttentionState {
        // Snapshot every mutable public field before validation. There is no
        // cached second representation: a later call snapshots again.
        let query_heads = AxisExtent::<QueryHead>::new(self.n_q_heads);
        let kv_heads = AxisExtent::<KvHead>::new(self.n_kv_heads);
        let head_dim = AxisExtent::<HeadDim>::new(self.head_dim);
        let rope_cfg = self.rope_cfg;
        let wq_tensor = self.wq.clone();
        let wk_tensor = self.wk.clone();
        let wv_tensor = self.wv.clone();
        let wo_tensor = self.wo.clone();
        let q_norm_tensor = self.q_norm.clone();
        let k_norm_tensor = self.k_norm.clone();

        assert!(
            kv_heads.get() > 0,
            "Qwen3Attention typed: n_kv_heads must be positive"
        );
        assert!(
            query_heads.get() > 0,
            "Qwen3Attention typed: n_q_heads must be positive"
        );
        assert!(
            query_heads.get() % kv_heads.get() == 0,
            "Qwen3Attention typed: n_q_heads must be divisible by n_kv_heads"
        );
        assert!(
            head_dim.get() > 0 && head_dim.get() % 2 == 0,
            "Qwen3Attention typed: head_dim must be positive and even"
        );

        let query_width = query_heads
            .checked_merge(head_dim)
            .expect("Qwen3Attention typed: query head extent overflow");
        let kv_width = kv_heads
            .checked_merge(head_dim)
            .expect("Qwen3Attention typed: KV head extent overflow");

        let wq_shape = wq_tensor.shape().0;
        assert_eq!(
            wq_shape.len(),
            2,
            "Qwen3Attention typed: wq must have shape [D,Hq*Dh]"
        );
        let hidden = AxisExtent::<Hidden>::new(wq_shape[0]);
        assert_eq!(
            wk_tensor.shape().0.len(),
            2,
            "Qwen3Attention typed: wk must have shape [D,Hkv*Dh]"
        );
        assert_eq!(
            wv_tensor.shape().0.len(),
            2,
            "Qwen3Attention typed: wv must have shape [D,Hkv*Dh]"
        );
        assert_eq!(
            wo_tensor.shape().0.len(),
            2,
            "Qwen3Attention typed: wo must have shape [Hq*Dh,D]"
        );
        assert_eq!(
            q_norm_tensor.shape().0.len(),
            1,
            "Qwen3Attention typed: q_norm must have shape [Dh]"
        );
        assert_eq!(
            k_norm_tensor.shape().0.len(),
            1,
            "Qwen3Attention typed: k_norm must have shape [Dh]"
        );

        let wq = QueryProjectionWeight::from_proven_axes(wq_tensor);
        let wk = KvProjectionWeight::from_proven_axes(wk_tensor);
        let wv = KvProjectionWeight::from_proven_axes(wv_tensor);
        let wo = OutputProjectionWeight::from_proven_axes(wo_tensor);
        let q_norm = HeadScale::from_proven_axes(q_norm_tensor);
        let k_norm = HeadScale::from_proven_axes(k_norm_tensor);
        assert_eq!(
            wq.flattened_query_extent(),
            query_width,
            "Qwen3Attention typed: wq must have shape [D,Hq*Dh]"
        );
        assert_eq!(
            wk.hidden_extent(),
            hidden,
            "Qwen3Attention typed: wk must have shape [D,Hkv*Dh]"
        );
        assert_eq!(
            wk.flattened_kv_extent(),
            kv_width,
            "Qwen3Attention typed: wk must have shape [D,Hkv*Dh]"
        );
        assert_eq!(
            wv.hidden_extent(),
            hidden,
            "Qwen3Attention typed: wv must have shape [D,Hkv*Dh]"
        );
        assert_eq!(
            wv.flattened_kv_extent(),
            kv_width,
            "Qwen3Attention typed: wv must have shape [D,Hkv*Dh]"
        );
        assert_eq!(
            wo.flattened_query_extent(),
            query_width,
            "Qwen3Attention typed: wo must have shape [Hq*Dh,D]"
        );
        assert_eq!(
            wo.hidden_extent(),
            hidden,
            "Qwen3Attention typed: wo must have shape [Hq*Dh,D]"
        );
        assert_eq!(
            q_norm.head_dim_extent(),
            head_dim,
            "Qwen3Attention typed: q_norm must have shape [Dh]"
        );
        assert_eq!(
            k_norm.head_dim_extent(),
            head_dim,
            "Qwen3Attention typed: k_norm must have shape [Dh]"
        );

        ValidatedAttentionState {
            geometry: AttentionGeometry {
                hidden,
                query_heads,
                kv_heads,
                head_dim,
                query_width,
                kv_width,
            },
            rope_cfg,
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
        }
    }

    /// Typed ordinary Qwen3 attention over one complete sequence.
    ///
    /// The type signatures expose each mathematical axis transformation while
    /// the concrete operations below intentionally mirror the independent
    /// legacy derivation in [`Self::forward`].
    pub fn forward_typed(&self, x: &FullHiddenStates) -> FullHiddenStates {
        let state = self.validated_attention_state();
        assert_eq!(
            x.hidden_extent(),
            state.geometry.hidden,
            "Qwen3Attention typed: hidden dimension mismatch"
        );
        let sequence = x.sequence_extent().get();
        if sequence > 0 {
            state
                .rope_cfg
                .start_pos
                .checked_add(sequence - 1)
                .expect("Qwen3Attention typed: RoPE position overflow");
        }

        // 1. X[B,T,D] is projected into the three distinct head spaces.
        let projected_queries = linear::project_queries(x, &state.wq, "q_proj");
        let projected_keys = linear::project_keys(x, &state.wk, "k_proj");
        let projected_values = linear::project_values(x, &state.wv, "v_proj");
        debug_assert_eq!(
            projected_queries.flattened_query_extent(),
            state.geometry.query_width
        );
        debug_assert_eq!(
            projected_keys.flattened_kv_extent(),
            state.geometry.kv_width
        );
        debug_assert_eq!(
            projected_values.flattened_kv_extent(),
            state.geometry.kv_width
        );

        // 2. The merged head coordinates become explicit tensor axes.
        let queries = shape_ops::split_query_heads(
            &projected_queries,
            state.geometry.query_heads,
            state.geometry.head_dim,
            "q_reshape",
        );
        let keys = shape_ops::split_kv_heads(
            &projected_keys,
            state.geometry.kv_heads,
            state.geometry.head_dim,
            "k_reshape",
        );
        let values = shape_ops::split_kv_heads(
            &projected_values,
            state.geometry.kv_heads,
            state.geometry.head_dim,
            "v_reshape",
        );

        // 3. Qwen3 normalizes each Q/K head before encoding position.
        let queries = norm::normalize_queries(&queries, &state.q_norm, "q_norm");
        let keys = norm::normalize_keys(&keys, &state.k_norm, "k_norm");

        // 4. Q and K receive the same complete-sequence position system.
        let queries = rope::rotate_queries(&queries, state.rope_cfg, "q_rope");
        let keys = rope::rotate_keys(&keys, state.rope_cfg, "k_rope");

        // 5. Causal grouped-query attention maps Q heads back to Q heads.
        let attended = gqa::gqa_attention_typed(&queries, &keys, &values, "gqa");

        // 6. Merge heads and return to model hidden space.
        let attended = shape_ops::merge_query_heads(&attended, "attn_reshape");
        linear::project_attention_output(&attended, &state.wo, "o_proj")
    }

    /// Typed context-parallel Qwen3 attention over equal contiguous shards.
    ///
    /// Slice order is the logical rank order. Local projections stay local;
    /// only the single typed CP-GQA operation observes all K/V shards.
    pub fn forward_context_parallel_typed(
        &self,
        x_shards: &[ShardHiddenStates],
    ) -> Vec<ShardHiddenStates> {
        assert!(
            !x_shards.is_empty(),
            "Qwen3Attention typed context parallel: at least one input shard is required"
        );
        let state = self.validated_attention_state();
        let batch = x_shards[0].batch_extent();
        let local_sequence = x_shards[0].sequence_extent();
        let hidden = x_shards[0].hidden_extent();
        assert!(
            local_sequence.get() > 0,
            "Qwen3Attention typed context parallel: local sequence length must be positive"
        );
        assert_eq!(
            hidden, state.geometry.hidden,
            "Qwen3Attention typed context parallel: hidden dimension mismatch"
        );
        for shard in x_shards {
            assert_eq!(
                (
                    shard.batch_extent(),
                    shard.sequence_extent(),
                    shard.hidden_extent(),
                ),
                (batch, local_sequence, hidden),
                "Qwen3Attention typed context parallel: every input shard must have the same shape"
            );
        }
        let layout = EqualContiguousAttentionLayout::new(x_shards.len(), local_sequence.get())
            .unwrap_or_else(|error| match error {
                AttentionLayoutError::GlobalSequenceOverflow { .. } => {
                    panic!("Qwen3Attention typed context parallel: global sequence length overflow")
                }
                _ => unreachable!("shard count and local sequence were validated"),
            });
        state
            .rope_cfg
            .start_pos
            .checked_add(layout.global_sequence() - 1)
            .expect("Qwen3Attention typed context parallel: RoPE start position overflow");

        let mut query_shards = Vec::with_capacity(x_shards.len());
        let mut key_shards = Vec::with_capacity(x_shards.len());
        let mut value_shards = Vec::with_capacity(x_shards.len());

        for (rank, hidden_states) in x_shards.iter().enumerate() {
            let prefix = format!("cp{rank}");

            // 1. Each rank projects only the hidden states it owns.
            let projected_queries =
                linear::project_queries(hidden_states, &state.wq, &format!("{prefix}.q_proj"));
            let projected_keys =
                linear::project_keys(hidden_states, &state.wk, &format!("{prefix}.k_proj"));
            let projected_values =
                linear::project_values(hidden_states, &state.wv, &format!("{prefix}.v_proj"));
            debug_assert_eq!(
                projected_queries.flattened_query_extent(),
                state.geometry.query_width
            );
            debug_assert_eq!(
                projected_keys.flattened_kv_extent(),
                state.geometry.kv_width
            );
            debug_assert_eq!(
                projected_values.flattened_kv_extent(),
                state.geometry.kv_width
            );

            // 2. Head axes and per-head Q/K normalization remain local.
            let queries = shape_ops::split_query_heads(
                &projected_queries,
                state.geometry.query_heads,
                state.geometry.head_dim,
                &format!("{prefix}.q_reshape"),
            );
            let keys = shape_ops::split_kv_heads(
                &projected_keys,
                state.geometry.kv_heads,
                state.geometry.head_dim,
                &format!("{prefix}.k_reshape"),
            );
            let values = shape_ops::split_kv_heads(
                &projected_values,
                state.geometry.kv_heads,
                state.geometry.head_dim,
                &format!("{prefix}.v_reshape"),
            );
            let queries =
                norm::normalize_queries(&queries, &state.q_norm, &format!("{prefix}.q_norm"));
            let keys = norm::normalize_keys(&keys, &state.k_norm, &format!("{prefix}.k_norm"));

            // 3. Local position i on rank r represents global r*S + i.
            let query_block = layout
                .query_block(rank)
                .expect("validated context-parallel rank");
            let start_pos = state
                .rope_cfg
                .start_pos
                .checked_add(query_block.start())
                .expect("Qwen3Attention typed context parallel: RoPE start position overflow");
            let rank_rope = rope::RopeConfig {
                base: state.rope_cfg.base,
                start_pos,
            };
            query_shards.push(rope::rotate_queries(
                &queries,
                rank_rope,
                &format!("{prefix}.q_rope"),
            ));
            key_shards.push(rope::rotate_keys(
                &keys,
                rank_rope,
                &format!("{prefix}.k_rope"),
            ));
            value_shards.push(values);
        }

        // 4. This is the one operation that crosses logical rank boundaries.
        let attended_shards = context_parallel_gqa::context_parallel_gqa_attention_typed(
            &query_shards,
            &key_shards,
            &value_shards,
            "context_parallel_gqa",
        );

        // 5. Every rank independently returns its result to hidden space.
        attended_shards
            .iter()
            .enumerate()
            .map(|(rank, attended)| {
                let merged =
                    shape_ops::merge_query_heads(attended, &format!("cp{rank}.attn_reshape"));
                linear::project_attention_output(&merged, &state.wo, &format!("cp{rank}.o_proj"))
            })
            .collect()
    }

    /// `x`: `[B, T, D]` → output `[B, T, D]`.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (b, t) = {
            let xv = x.inner.borrow();
            let sh = &xv.value.shape.0;
            assert_eq!(sh.len(), 3, "Qwen3Attention input must be [B,T,D]");
            (sh[0], sh[1])
        };
        let hq = self.n_q_heads;
        let hk = self.n_kv_heads;
        let dh = self.head_dim;

        // 1. Bias-free Q/K/V projections.
        let q_flat = linear::linear(x, &self.wq, "q_proj"); // [B,T,Hq*Dh]
        let k_flat = linear::linear(x, &self.wk, "k_proj"); // [B,T,Hk*Dh]
        let v_flat = linear::linear(x, &self.wv, "v_proj"); // [B,T,Hk*Dh]

        // 2. Reshape to multi-head: [B,T,H,Dh].
        let q4 = shape_ops::reshape(&q_flat, &[b, t, hq, dh], "q_reshape");
        let k4 = shape_ops::reshape(&k_flat, &[b, t, hk, dh], "k_reshape");
        let v4 = shape_ops::reshape(&v_flat, &[b, t, hk, dh], "v_reshape");

        // 3. Per-head Q/K RMSNorm (gamma is shared across heads, shape [Dh]).
        //    rms_norm reduces over the last axis, so [B,T,H,Dh] works directly.
        let q4n = norm::rms_norm(&q4, &self.q_norm, "q_norm");
        let k4n = norm::rms_norm(&k4, &self.k_norm, "k_norm");

        // 4. RoPE on Q and K (V is not rotated).
        let q4r = rope::rope(&q4n, self.rope_cfg, "q_rope");
        let k4r = rope::rope(&k4n, self.rope_cfg, "k_rope");

        // 5. GQA causal attention.
        let attn = gqa::gqa_attention(&q4r, &k4r, &v4, "gqa"); // [B,T,Hq,Dh]

        // 6. Flatten heads and bias-free output projection.
        let attn_flat = shape_ops::reshape(&attn, &[b, t, hq * dh], "attn_reshape");
        linear::linear(&attn_flat, &self.wo, "o_proj")
    }

    /// Logical context-parallel attention over equal contiguous token shards.
    ///
    /// Every non-attention operation runs independently on `[B,S,D]` for each
    /// logical rank. Q/K RoPE uses the rank's global contiguous token offset;
    /// context-parallel GQA then exchanges materialized K/V logically and
    /// returns one `[B,S,Hq,Dh]` output per rank. Weights remain shared tensors,
    /// so autograd sums their gradient contributions from every rank.
    pub fn forward_context_parallel(&self, x_shards: &[Tensor]) -> Vec<Tensor> {
        assert!(
            !x_shards.is_empty(),
            "Qwen3Attention context parallel: at least one input shard is required"
        );
        let expected_d = self.wq.inner.borrow().value.shape.0[0];
        let first_shape = x_shards[0].shape().0;
        assert_eq!(
            first_shape.len(),
            3,
            "Qwen3Attention context parallel: input must be [B,S,D]"
        );
        let (b, local_t, d) = (first_shape[0], first_shape[1], first_shape[2]);
        assert!(
            local_t > 0,
            "Qwen3Attention context parallel: local sequence length must be positive"
        );
        assert_eq!(
            d, expected_d,
            "Qwen3Attention context parallel: hidden dimension mismatch"
        );
        for shard in x_shards {
            assert_eq!(
                shard.shape().0.as_slice(),
                [b, local_t, d],
                "Qwen3Attention context parallel: every input shard must have the same shape"
            );
        }

        let layout =
            EqualContiguousAttentionLayout::new(x_shards.len(), local_t).unwrap_or_else(|error| {
                match error {
                    AttentionLayoutError::GlobalSequenceOverflow { .. } => {
                        panic!("Qwen3Attention context parallel: global sequence length overflow")
                    }
                    _ => unreachable!("shard count and local sequence were validated"),
                }
            });

        let hq = self.n_q_heads;
        let hk = self.n_kv_heads;
        let dh = self.head_dim;
        let mut q_shards = Vec::with_capacity(x_shards.len());
        let mut k_shards = Vec::with_capacity(x_shards.len());
        let mut v_shards = Vec::with_capacity(x_shards.len());

        for (rank, x) in x_shards.iter().enumerate() {
            let prefix = format!("cp{rank}");
            let q_flat = linear::linear(x, &self.wq, &format!("{prefix}.q_proj"));
            let k_flat = linear::linear(x, &self.wk, &format!("{prefix}.k_proj"));
            let v_flat = linear::linear(x, &self.wv, &format!("{prefix}.v_proj"));

            let q4 = shape_ops::reshape(
                &q_flat,
                &[b, local_t, hq, dh],
                &format!("{prefix}.q_reshape"),
            );
            let k4 = shape_ops::reshape(
                &k_flat,
                &[b, local_t, hk, dh],
                &format!("{prefix}.k_reshape"),
            );
            let v4 = shape_ops::reshape(
                &v_flat,
                &[b, local_t, hk, dh],
                &format!("{prefix}.v_reshape"),
            );
            let q4n = norm::rms_norm(&q4, &self.q_norm, &format!("{prefix}.q_norm"));
            let k4n = norm::rms_norm(&k4, &self.k_norm, &format!("{prefix}.k_norm"));
            let query_block = layout
                .query_block(rank)
                .expect("validated context-parallel rank");
            let rank_rope = rope::RopeConfig {
                base: self.rope_cfg.base,
                start_pos: self
                    .rope_cfg
                    .start_pos
                    .checked_add(query_block.start())
                    .expect("Qwen3Attention context parallel: RoPE start position overflow"),
            };
            q_shards.push(rope::rope(&q4n, rank_rope, &format!("{prefix}.q_rope")));
            k_shards.push(rope::rope(&k4n, rank_rope, &format!("{prefix}.k_rope")));
            v_shards.push(v4);
        }

        let attention_shards = context_parallel_gqa::context_parallel_gqa_attention(
            &q_shards,
            &k_shards,
            &v_shards,
            "context_parallel_gqa",
        );
        attention_shards
            .iter()
            .enumerate()
            .map(|(rank, attention)| {
                let flattened = shape_ops::reshape(
                    attention,
                    &[b, local_t, hq * dh],
                    &format!("cp{rank}.attn_reshape"),
                );
                linear::linear(&flattened, &self.wo, &format!("cp{rank}.o_proj"))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// SwiGLU MLP
// ---------------------------------------------------------------------------

pub struct Qwen3MLP {
    pub gate_proj: Tensor, // [D, F]
    pub up_proj: Tensor,   // [D, F]
    pub down_proj: Tensor, // [F, D]
}

impl Qwen3MLP {
    pub fn new(cfg: &Qwen3Config) -> Self {
        let d = cfg.hidden_size;
        let f = cfg.intermediate_size;
        let scale = (d as f32).sqrt().recip() * 0.5;
        Self {
            gate_proj: param(d, f, scale),
            up_proj: param(d, f, scale),
            down_proj: param(f, d, (f as f32).sqrt().recip() * 0.5),
        }
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.gate_proj, &self.up_proj, &self.down_proj]
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let gate = linear::linear(x, &self.gate_proj, "gate_proj");
        let up = linear::linear(x, &self.up_proj, "up_proj");
        let gated = activations::silu(&gate, "silu");
        let mixed = basic::mul(&gated, &up, "swiglu_mul");
        linear::linear(&mixed, &self.down_proj, "down_proj")
    }
}

// ---------------------------------------------------------------------------
// Decoder block
// ---------------------------------------------------------------------------

pub struct Qwen3Block {
    pub input_layernorm: Tensor,          // [D]  (RMSNorm gamma)
    pub post_attention_layernorm: Tensor, // [D]
    pub self_attn: Qwen3Attention,
    pub mlp: Qwen3MLP,
}

impl Qwen3Block {
    pub fn new(cfg: &Qwen3Config) -> Self {
        Self {
            input_layernorm: ones(cfg.hidden_size),
            post_attention_layernorm: ones(cfg.hidden_size),
            self_attn: Qwen3Attention::new(cfg),
            mlp: Qwen3MLP::new(cfg),
        }
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut p = vec![&self.input_layernorm, &self.post_attention_layernorm];
        p.extend(self.self_attn.parameters());
        p.extend(self.mlp.parameters());
        p
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Pre-norm self-attention.
        let h = norm::rms_norm(x, &self.input_layernorm, "input_layernorm");
        let a = self.self_attn.forward(&h);
        let x = basic::add(x, &a, "attn_residual");

        // Pre-norm SwiGLU MLP.
        let h = norm::rms_norm(
            &x,
            &self.post_attention_layernorm,
            "post_attention_layernorm",
        );
        let m = self.mlp.forward(&h);
        basic::add(&x, &m, "mlp_residual")
    }
}

// ---------------------------------------------------------------------------
// Full causal LM
// ---------------------------------------------------------------------------

pub struct Qwen3Model {
    pub cfg: Qwen3Config,
    pub embed_tokens: Tensor, // [V, D]
    pub layers: Vec<Qwen3Block>,
    pub final_norm: Tensor, // [D]
    /// LM head `[D, V]`.  When `tie_word_embeddings` is true this stores the
    /// untied head transposed from `embed_tokens` at construction; callers
    /// that want true weight sharing should pass `&embed_tokens.T` into
    /// `cross_entropy` themselves.
    pub lm_head: Tensor,
}

impl Qwen3Model {
    pub fn new(cfg: Qwen3Config) -> Self {
        let v = cfg.vocab_size;
        let d = cfg.hidden_size;
        let embed_scale = (d as f32).sqrt().recip() * 0.5;
        let embed_tokens = param(v, d, embed_scale);
        let final_norm = ones(d);
        let lm_head = param(d, v, embed_scale);
        let layers = (0..cfg.num_hidden_layers)
            .map(|_| Qwen3Block::new(&cfg))
            .collect();
        Self {
            cfg,
            embed_tokens,
            layers,
            final_norm,
            lm_head,
        }
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut p = vec![&self.embed_tokens, &self.final_norm, &self.lm_head];
        for layer in &self.layers {
            p.extend(layer.parameters());
        }
        p
    }

    /// Run the model on token IDs `[B,T]`.  Returns logits `[B,T,V]`.
    pub fn forward(&self, ids: &[usize], b: usize, t: usize) -> Tensor {
        let mut h = embedding::embedding(ids, b, t, &self.embed_tokens, "embed_tokens");
        for (li, layer) in self.layers.iter().enumerate() {
            // Use the block's own internal op names; layer index recorded via
            // the OpCall debug stream if needed.
            let _ = li;
            h = layer.forward(&h);
        }
        let h = norm::rms_norm(&h, &self.final_norm, "final_norm");
        linear::linear(&h, &self.lm_head, "lm_head")
    }
}
